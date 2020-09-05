import math

import torch
import torch.nn as nn
from npf.utils.helpers import ProbabilityConverter, backward_pdb, mask_and_apply
from npf.utils.initialization import init_param_, weights_init
from torch.nn import functional as F

from .mlp import MLP

__all__ = ["SetConv", "MlpRBF", "ExpRBF", "UnsharedExpRBF"]


class UnsharedExpRBF(nn.Module):
    """Gaussian radial basis function.

    Parameters
    ----------
    x_dim : int
        Dimensionality of input. Placeholder (not used)

    max_dist : float, optional
        Max distance between the closest query and target, used for intialisation.

    max_dist_weight : float, optional
        Weight that should be given to a maximum distance. Note that min_dist_weight
        is 1, so can also be seen as a ratio.

    p : int, optional
        p-norm to use, If p=2, exponential quadratic => Gaussian.

    is_share_sigma : bool, optional
        Whether to share sigma between the density channel and the weight.
    """

    def __init__(
        self,
        x_dim,
        max_dist=1 / 256,
        max_dist_weight=0.99,
        p=2,
        **kwargs
    ):
        super().__init__()

        self.max_dist = max_dist
        self.max_dist_weight = max_dist_weight
        self.length_scale_param = nn.Parameter(torch.tensor([0.0]*2))
        self.p = p
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)
        # set the parameter depending on the weight to give to a maxmum distance
        # query. i.e. exp(- (max_dist / sigma).pow(p)) = max_dist_weight
        # => sigma = max_dist / ((- log(max_dist_weight))**(1/p))
        max_dist_sigma = self.max_dist / (
            (-math.log(self.max_dist_weight)) ** (1 / self.p)
        )
        # inverse_softplus : log(exp(y) - 1)
        max_dist_param = math.log(math.exp(max_dist_sigma) - 1)
        self.length_scale_param = nn.Parameter(torch.tensor([max_dist_param]*2))

    def forward(self, diff):

        # size=[batch_size, n_keys, n_queries, 1]
        dist = torch.norm(diff, p=self.p, dim=-1, keepdim=True)

        # compute exponent making sure no division by 0
        sigma = 1e-5 + F.softplus(self.length_scale_param)

        inp = -(dist / sigma).pow(self.p)

        # size=[batch_size, n_keys, n_queries, 1]
        out = torch.exp(inp)
        
        # size=[batch_size, n_keys, 1]
        density = out[...,1:].sum(dim=-2)

        # size=[batch_size, n_keys, n_queries, 1]
        out = out[...,0:1] / (density.unsqueeze(2) + 1e-8)

        return out, density


class ExpRBF(nn.Module):
    """Exponential radial basis function.

    Parameters
    ----------
    x_dim : int
        Dimensionality of input. Placeholder (not used)

    max_dist : float, optional
        Max distance between the closest query and target, used for intialisation.

    max_dist_weight : float, optional
        Weight that should be given to a maximum distance. Note that min_dist_weight
        is 1, so can also be seen as a ratio.

    p : int, optional
        p-norm to use, If p=2, exponential quadratic => Gaussian.
    """

    def __init__(self, x_dim, max_dist=1 / 256, max_dist_weight=0.9, p=2, **kwargs):
        super().__init__()

        self.max_dist = max_dist
        self.max_dist_weight = max_dist_weight
        self.length_scale_param = nn.Parameter(torch.tensor([0.0]))
        self.p = p
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)
        # set the parameter depending on the weight to give to a maxmum distance
        # query. i.e. exp(- (max_dist / sigma).pow(p)) = max_dist_weight
        # => sigma = max_dist / ((- log(max_dist_weight))**(1/p))
        max_dist_sigma = self.max_dist / (
            (-math.log(self.max_dist_weight)) ** (1 / self.p)
        )
        # inverse_softplus : log(exp(y) - 1)
        max_dist_param = math.log(math.exp(max_dist_sigma) - 1)
        self.length_scale_param = nn.Parameter(torch.tensor([max_dist_param]))

    def forward(self, diff):

        # size=[batch_size, n_keys, n_queries, kq_size]
        dist = torch.norm(diff, p=self.p, dim=-1, keepdim=True)

        # compute exponent making sure no division by 0
        sigma = 1e-5 + F.softplus(self.length_scale_param)

        inp = -(dist / sigma).pow(self.p)
        out = torch.softmax(
            inp, dim=-2
        )  # numerically stable normalization of the weights by density

        # size=[batch_size, n_keys, kq_size]
        density = torch.exp(inp).sum(dim=-2)

        return out, density


class MlpRBF(nn.Module):
    """Gaussian radial basis function.

    Parameters
    ----------
    x_dim : int
        Number of spatio-temporal input dimensions.

    is_abs_dist : bool, optional
        Whether to force the kernel to be symmetric around 0.

    window_size : int, bool
        Maximimum distance to consider

    kwargs :
        Placeholder
    """

    def __init__(self, x_dim, is_abs_dist=True, window_size=0.25, **kwargs):
        super().__init__()
        self.is_abs_dist = is_abs_dist
        self.window_size = window_size
        self.mlp = MLP(x_dim, 1, n_hidden_layers=3, hidden_size=16)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, diff):
        abs_diff = diff.abs()

        # select only points with distance less than window_size (for extrapolation + speed)
        mask = abs_diff < self.window_size

        if self.is_abs_dist:
            diff = abs_diff

        # sparse operation (apply only on mask) => 2-3x speedup
        weight = mask_and_apply(
            diff, mask, lambda x: self.mlp(x.unsqueeze(1)).abs().squeeze()
        )
        weight = weight * mask.float()  # set to 0 points that are further than windo

        density = weight.sum(dim=-2, keepdim=True)
        out = weight / (density + 1e-5)  # don't divide by 0

        return out, density.squeeze(-1)


class SetConv(nn.Module):
    """Applies a convolution over a set of inputs, i.e. generalizes `nn._ConvNd`
    to non uniformly sampled samples [1].

    Parameters
    ----------
    x_dim : int
        Number of spatio-temporal dimensions of input.

    in_channels : int
        Number of input channels.

    out_channels : int
        Number of output channels.

    RadialBasisFunc : callable, optional
        Function which returns the "weight" of each points as a function of their
        distance (i.e. for usual CNN that would be the filter).

    kwargs :
        Additional arguments to `RadialBasisFunc`.

    References
    ----------
    [1] Gordon, Jonathan, et al. "Convolutional conditional neural processes." arXiv preprint
    arXiv:1910.13556 (2019).
    """

    def __init__(
        self, x_dim, in_channels, out_channels, RadialBasisFunc=ExpRBF, **kwargs
    ):
        super().__init__()
        assert x_dim == 1, "Currently only supports single spatial dimension `x_dim==1`"
        self.radial_basis_func = RadialBasisFunc(x_dim, **kwargs)
        self.resizer = nn.Linear(in_channels + 1, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, keys, queries, values):
        """
        Compute the set convolution between {key, value} and {query}.

        TODO
        ----
        - should sort the keys and queries to not compute differences if outside
        of given receptive field (large memory savings).

        Parameters
        ----------
        keys : torch.Tensor, size=[batch_size, n_keys, kq_size]
        queries : torch.Tensor, size=[batch_size, n_queries, kq_size]
        values : torch.Tensor, size=[batch_size, n_keys, in_channels]

        Return
        ------
        targets : torch.Tensor, size=[batch_size, n_queries, out_channels]
        """
        # prepares for broadcasted computations
        keys = keys.unsqueeze(1)
        queries = queries.unsqueeze(2)
        values = values.unsqueeze(1)

        # weight size = [batch_size, n_queries, n_keys, 1]
        # density size = [batch_size, n_queries, 1]
        weight, density = self.radial_basis_func(keys - queries)

        # size = [batch_size, n_queries, value_size]
        targets = (weight * values).sum(dim=2)

        # size = [batch_size, n_queries, value_size+1]
        targets = torch.cat([targets, density], dim=-1)

        return self.resizer(targets)
