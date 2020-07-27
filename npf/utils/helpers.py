import operator
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rv_discrete
from torch.distributions import Normal
from torch.distributions.independent import Independent

from .initialization import weights_init


def sum_from_nth_dim(t, dim):
    """Sum all dims from `dim`. E.g. sum_after_nth_dim(torch.rand(2,3,4,5), 2).shape = [2,3]"""
    return t.view(*t.shape[:dim], -1).sum(-1)


def logcumsumexp(x, dim):
    """Numerically stable log cumsum exp. SLow workaround waiting for https://github.com/pytorch/pytorch/pull/36308"""

    if (dim != -1) or (dim != x.ndimension() - 1):
        x = x.transpose(dim, -1)

    out = []
    for i in range(1, x.size(-1) + 1):
        out.append(torch.logsumexp(x[..., :i], dim=-1, keepdim=True))
    out = torch.cat(out, dim=-1)

    if (dim != -1) or (dim != x.ndimension() - 1):
        out = out.transpose(-1, dim)
    return out


class LightTailPareto(rv_discrete):
    def _cdf(self, k, alpha):
        # alpha is factor like in SUMO paper
        # m is minimum number of samples
        m = self.a  # lower bound of support

        # in the paper they us P(K >= k) but cdf is P(K <= k) = 1 - P(K > k) = 1 - P(K >= k + 1)
        k = k + 1

        # make sure has at least m samples
        k = np.clip(k - m, a_min=1, a_max=None)  # makes sure no division by 0
        alpha = alpha - m

        # sample using pmf 1/k but with finite expectation
        cdf = 1 - np.where(k < alpha, 1 / k, (1 / alpha) * (0.9) ** (k - alpha))

        return cdf


def isin_range(x, valid_range):
    """Check if array / tensor is in a given range elementwise."""
    return ((x >= valid_range[0]) & (x <= valid_range[1])).all()


def channels_to_2nd_dim(X):
    """
    Takes a signal with channels on the last dimension (for most operations) and
    returns it with channels on the second dimension (for convolutions).
    """
    return X.permute(*([0, X.dim() - 1] + list(range(1, X.dim() - 1))))


def channels_to_last_dim(X):
    """
    Takes a signal with channels on the second dimension (for convolutions) and
    returns it with channels on the last dimension (for most operations).
    """
    return X.permute(*([0] + list(range(2, X.dim())) + [1]))


def mask_and_apply(x, mask, f):
    """Applies a callable on a masked version of a input."""
    tranformed_selected = f(x.masked_select(mask))
    return x.masked_scatter(mask, tranformed_selected)


def indep_shuffle_(a, axis=-1):
    """
    Shuffle `a` in-place along the given axis.

    Apply `numpy.random.shuffle` to the given axis of `a`.
    Each one-dimensional slice is shuffled independently.

    Credits : https://github.com/numpy/numpy/issues/5173
    """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])


def ratio_to_int(percentage, max_val):
    """Converts a ratio to an integer if it is smaller than 1."""
    if 1 <= percentage <= max_val:
        out = percentage
    elif 0 <= percentage < 1:
        out = percentage * max_val
    else:
        raise ValueError("percentage={} outside of [0,{}].".format(percentage, max_val))

    return int(out)


def prod(iterable):
    """Compute the product of all elements in an iterable."""
    return reduce(operator.mul, iterable, 1)


def rescale_range(X, old_range, new_range):
    """Rescale X linearly to be in `new_range` rather than `old_range`."""
    old_min = old_range[0]
    new_min = new_range[0]
    old_delta = old_range[1] - old_min
    new_delta = new_range[1] - new_min
    return (((X - old_min) * new_delta) / old_delta) + new_min


def MultivariateNormalDiag(loc, scale_diag):
    """Multi variate Gaussian with a diagonal covariance function (on the last dimension)."""
    if loc.dim() < 1:
        raise ValueError("loc must be at least one-dimensional.")
    return Independent(Normal(loc, scale_diag), 1)


def clamp(
    x,
    minimum=-float("Inf"),
    maximum=float("Inf"),
    is_leaky=False,
    negative_slope=0.01,
    hard_min=None,
    hard_max=None,
):
    """
    Clamps a tensor to the given [minimum, maximum] (leaky) bound, with
    an optional hard clamping.
    """
    lower_bound = (
        (minimum + negative_slope * (x - minimum))
        if is_leaky
        else torch.zeros_like(x) + minimum
    )
    upper_bound = (
        (maximum + negative_slope * (x - maximum))
        if is_leaky
        else torch.zeros_like(x) + maximum
    )
    clamped = torch.max(lower_bound, torch.min(x, upper_bound))

    if hard_min is not None or hard_max is not None:
        if hard_min is None:
            hard_min = -float("Inf")
        elif hard_max is None:
            hard_max = float("Inf")
        clamped = clamp(x, minimum=hard_min, maximum=hard_max, is_leaky=False)

    return clamped


class ProbabilityConverter(nn.Module):
    """Maps floats to probabilites (between 0 and 1), element-wise.

    Parameters
    ----------
    min_p : float, optional
        Minimum probability, can be useful to set greater than 0 in order to keep
        gradient flowing if the probability is used for convex combinations of
        different parts of the model. Note that maximum probability is `1-min_p`.

    activation : {"sigmoid", "hard-sigmoid", "leaky-hard-sigmoid"}, optional
        name of the activation to use to generate the probabilities. `sigmoid`
        has the advantage of being smooth and never exactly 0 or 1, which helps
        gradient flows. `hard-sigmoid` has the advantage of making all values
        between min_p and max_p equiprobable.

    is_train_temperature : bool, optional
        Whether to train the paremeter controling the steapness of the activation.
        This is useful when x is used for multiple tasks, and you don't want to
        constraint its magnitude.

    is_train_bias : bool, optional
        Whether to train the bias to shift the activation. This is useful when x is
        used for multiple tasks, and you don't want to constraint it's scale.

    trainable_dim : int, optional
        Size of the trainable bias and termperature. If `1` uses the same vale
        across all dimension, if not should be equal to the number of input
        dimensions to different trainable aprameters for each dimension. Note
        that the iitial value will still be the same for all dimensions.

    initial_temperature : int, optional
        Initial temperature, a higher temperature makes the activation steaper.

    initial_probability : float, optional
        Initial probability you want to start with.

    initial_x : float, optional
        First value that will be given to the function, important to make
        `initial_probability` work correctly.

    bias_transformer : callable, optional
        Transformer function of the bias. This function should only take care of
        the boundaries (e.g. leaky relu or relu).

    temperature_transformer : callable, optional
        Transformer function of the temperature. This function should only take
        care of the boundaries (e.g. leaky relu  or relu).
    """

    def __init__(
        self,
        min_p=0.0,
        activation="sigmoid",
        is_train_temperature=False,
        is_train_bias=False,
        trainable_dim=1,
        initial_temperature=1.0,
        initial_probability=0.5,
        initial_x=0,
        bias_transformer=nn.Identity(),
        temperature_transformer=nn.Identity(),
    ):

        super().__init__()
        self.min_p = min_p
        self.activation = activation
        self.is_train_temperature = is_train_temperature
        self.is_train_bias = is_train_bias
        self.trainable_dim = trainable_dim
        self.initial_temperature = initial_temperature
        self.initial_probability = initial_probability
        self.initial_x = initial_x
        self.bias_transformer = bias_transformer
        self.temperature_transformer = temperature_transformer

        self.reset_parameters()

    def reset_parameters(self):
        self.temperature = torch.tensor([self.initial_temperature] * self.trainable_dim)
        if self.is_train_temperature:
            self.temperature = nn.Parameter(self.temperature)

        initial_bias = self._probability_to_bias(
            self.initial_probability, initial_x=self.initial_x
        )

        self.bias = torch.tensor([initial_bias] * self.trainable_dim)
        if self.is_train_bias:
            self.bias = nn.Parameter(self.bias)

    def forward(self, x):
        self.temperature.to(x.device)
        self.bias.to(x.device)

        temperature = self.temperature_transformer(self.temperature)
        bias = self.bias_transformer(self.bias)

        if self.activation == "sigmoid":
            full_p = torch.sigmoid((x + bias) * temperature)

        elif self.activation in ["hard-sigmoid", "leaky-hard-sigmoid"]:
            # uses 0.2 and 0.5 to be similar to sigmoid
            y = 0.2 * ((x + bias) * temperature) + 0.5

            if self.activation == "leaky-hard-sigmoid":
                full_p = clamp(
                    y,
                    minimum=0.1,
                    maximum=0.9,
                    is_leaky=True,
                    negative_slope=0.01,
                    hard_min=0,
                    hard_max=0,
                )
            elif self.activation == "hard-sigmoid":
                full_p = clamp(y, minimum=0.0, maximum=1.0, is_leaky=False)

        else:
            raise ValueError("Unkown activation : {}".format(self.activation))

        p = rescale_range(full_p, (0, 1), (self.min_p, 1 - self.min_p))

        return p

    def _probability_to_bias(self, p, initial_x=0):
        """Compute the bias to use to satisfy the constraints."""
        assert p > self.min_p and p < 1 - self.min_p
        range_p = 1 - self.min_p * 2
        p = (p - self.min_p) / range_p
        p = torch.tensor(p, dtype=torch.float)

        if self.activation == "sigmoid":
            bias = -(torch.log((1 - p) / p) / self.initial_temperature + initial_x)

        elif self.activation in ["hard-sigmoid", "leaky-hard-sigmoid"]:
            bias = ((p - 0.5) / 0.2) / self.initial_temperature - initial_x

        return bias


def dist_to_device(dist, device):
    """Set a distirbution to a given device."""
    if dist is None:
        return
    dist.base_dist.loc = dist.base_dist.loc.to(device)
    dist.base_dist.scale = dist.base_dist.loc.to(device)


def make_abs_conv(Conv):
    """Make a convolution have only positive parameters."""

    class AbsConv(Conv):
        def forward(self, input):
            return F.conv2d(
                input,
                self.weight.abs(),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    return AbsConv


def make_padded_conv(Conv, Padder):
    """Make a convolution have any possible padding."""

    class PaddedConv(Conv):
        def __init__(self, *args, Padder=Padder, padding=0, **kwargs):
            old_padding = 0
            if Padder is None:
                Padder = nn.Identity
                old_padding = padding

            super().__init__(*args, padding=old_padding, **kwargs)
            self.padder = Padder(padding)

        def forward(self, X):
            X = self.padder(X)
            return super().forward(X)

    return PaddedConv


def make_depth_sep_conv(Conv):
    """Make a convolution module depth separable."""

    class DepthSepConv(nn.Module):
        """Make a convolution depth separable.

        Parameters
        ----------
        in_channels : int
            Number of input channels.

        out_channels : int
            Number of output channels.

        kernel_size : int

        **kwargs :
            Additional arguments to `Conv`
        """

        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            confidence=False,
            bias=True,
            **kwargs
        ):
            super().__init__()
            self.depthwise = Conv(
                in_channels,
                in_channels,
                kernel_size,
                groups=in_channels,
                bias=bias,
                **kwargs
            )
            self.pointwise = Conv(in_channels, out_channels, 1, bias=bias)
            self.reset_parameters()

        def forward(self, x):
            out = self.depthwise(x)
            out = self.pointwise(out)
            return out

        def reset_parameters(self):
            weights_init(self)

    return DepthSepConv


class CircularPad2d(nn.Module):
    """Implements a 2d circular padding."""

    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, (self.padding,) * 4, mode="circular")


class BackwardPDB(torch.autograd.Function):
    """Run PDB in the backward pass."""

    @staticmethod
    def forward(ctx, input, name="debugger"):
        ctx.name = name
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        if not torch.isfinite(grad_output).all() or not torch.isfinite(input).all():
            import pdb

            pdb.set_trace()
        return grad_output, None  # 2 args so return None for `name`


backward_pdb = BackwardPDB.apply
