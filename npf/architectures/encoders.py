import torch
import torch.nn as nn

from npf.utils.helpers import mask_and_apply
from npf.utils.initialization import weights_init

from .mlp import MLP

__all__ = [
    "RelativeSinusoidalEncodings",
    "SinusoidalEncodings",
    "merge_flat_input",
    "discard_ith_arg",
]


class SinusoidalEncodings(nn.Module):
    """
    Converts a batch of N-dimensional spatial input X with values between `[-1,1]`
    to a batch of flat in vector splitted in N subvectors that encode the position via
    sinusoidal encodings.

    Parameters
    ----------
    x_dim : int
        Number of spatial inputs.

    out_dim : int
        size of output encoding. Each x_dim will have an encoding of size
        `out_dim//x_dim`.
    """

    def __init__(self, x_dim, out_dim):
        super().__init__()
        self.x_dim = x_dim
        # dimension of encoding for eacg x dimension
        self.sub_dim = out_dim // self.x_dim
        # in "attention is all you need" used 10000 but 512 dim, try to keep the
        # same ratio regardless of dim
        self._C = 10000 * (self.sub_dim / 512) ** 2

        if out_dim % x_dim != 0:
            raise ValueError(
                "out_dim={} has to be dividable by x_dim={}.".format(out_dim, x_dim)
            )
        if self.sub_dim % 2 != 0:
            raise ValueError(
                "sum_dim=out_dim/x_dim={} has to be dividable by 2.".format(
                    self.sub_dim
                )
            )

        self._precompute_denom()

    def _precompute_denom(self):
        two_i_d = torch.arange(0, self.sub_dim, 2, dtype=torch.float) / self.sub_dim
        denom = torch.pow(self._C, two_i_d)
        denom = torch.repeat_interleave(denom, 2).unsqueeze(0)
        self.denom = denom.expand(1, self.x_dim, self.sub_dim)

    def forward(self, x):
        shape = x.shape
        # flatten besides last dim
        x = x.view(-1, shape[-1])
        # will only be passed once to GPU because precomputed
        self.denom = self.denom.to(x.device)
        # put x in a range which is similar to positions in NLP [1,51]
        x = (x.unsqueeze(-1) + 1) * 25 + 1
        out = x / self.denom
        out[:, :, 0::2] = torch.sin(out[:, :, 0::2])
        out[:, :, 1::2] = torch.cos(out[:, :, 1::2])
        # concatenate all different sinusoidal encodings for each x_dim
        # and unflatten
        out = out.view(*shape[:-1], self.sub_dim * self.x_dim)
        return out


class RelativeSinusoidalEncodings(nn.Module):
    """Return relative positions of inputs between [-1,1]."""

    def __init__(self, x_dim, out_dim, window_size=2):
        super().__init__()
        self.pos_encoder = SinusoidalEncodings(x_dim, out_dim)
        self.weight = nn.Linear(out_dim, out_dim, bias=False)
        self.window_size = window_size
        self.out_dim = out_dim

    def forward(self, keys_pos, queries_pos):
        # size=[batch_size, n_queries, n_keys, x_dim]
        diff = (keys_pos.unsqueeze(1) - queries_pos.unsqueeze(2)).abs()

        # the abs differences will be between between 0, self.window_size
        # we multipl by 2/self.window_size then remove 1 to  be [-1,1] which is
        # the range for `SinusoidalEncodings`
        scaled_diff = diff * 2 / self.window_size - 1
        out = self.weight(self.pos_encoder(scaled_diff))

        # set to 0 points that are further than window for extap
        out = out * (diff < self.window_size).float()

        return out


# META ENCODERS
class DiscardIthArg(nn.Module):
    """
    Helper module which discard the i^th argument of the constructor and forward,
    before being given to `To`.
    """

    def __init__(self, *args, i=0, To=nn.Identity, **kwargs):
        super().__init__()
        self.i = i
        self.destination = To(*self.filter_args(*args), **kwargs)

    def filter_args(self, *args):
        return [arg for i, arg in enumerate(args) if i != self.i]

    def forward(self, *args, **kwargs):
        return self.destination(*self.filter_args(*args), **kwargs)


def discard_ith_arg(module, i, **kwargs):
    def discarded_arg(*args, **kwargs2):
        return DiscardIthArg(*args, i=i, To=module, **kwargs, **kwargs2)

    return discarded_arg


class MergeFlatInputs(nn.Module):
    """
    Extend a module to take 2 flat inputs. It simply returns
    the concatenated flat inputs to the module `module({x1; x2})`.

    Parameters
    ----------
    FlatModule: nn.Module
        Module which takes a non flat inputs.

    x1_dim: int
        Dimensionality of the first flat inputs.

    x2_dim: int
        Dimensionality of the second flat inputs.

    n_out: int
        Size of ouput.

    is_sum_merge : bool, optional
        Whether to transform `flat_input` by an MLP first (if need to resize),
        then sum to `X` (instead of concatenating): useful if the difference in
        dimension between both inputs is very large => don't want one layer to
        depend only on a few dimension of a large input.

    kwargs:
        Additional arguments to FlatModule.
    """

    def __init__(self, FlatModule, x1_dim, x2_dim, n_out, is_sum_merge=False, **kwargs):
        super().__init__()
        self.is_sum_merge = is_sum_merge

        if self.is_sum_merge:
            dim = x1_dim
            self.resizer = MLP(x2_dim, dim)  # transform to be the correct size
        else:
            dim = x1_dim + x2_dim

        self.flat_module = FlatModule(dim, n_out, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, x1, x2):
        if self.is_sum_merge:
            x2 = self.resizer(x2)
            # use activation because if not 2 linear layers in a row => useless computation
            out = torch.relu(x1 + x2)
        else:
            out = torch.cat((x1, x2), dim=-1)

        return self.flat_module(out)


def merge_flat_input(module, is_sum_merge=False, **kwargs):
    """
    Extend a module to accept an additional flat input. I.e. the output should
    be called by `merge_flat_input(module)(x_shape, flat_dim, n_out, **kwargs)`.

    Notes
    -----
    - if x_shape is an integer (currently only available option), it simply returns
    the concatenated flat inputs to the module `module({x; flat_input})`.
    - if `is_sum_merge` then transform `flat_input` by an MLP first, then sum
    to `X` (instead of concatenating): useful if the difference in dimension
    between both inputs is very large => don't want one layer to depend only on
    a few dimension of a large input.
    """

    def merged_flat_input(x_shape, flat_dim, n_out, **kwargs2):
        assert isinstance(x_shape, int)
        return MergeFlatInputs(
            module,
            x_shape,
            flat_dim,
            n_out,
            is_sum_merge=is_sum_merge,
            **kwargs2,
            **kwargs
        )

    return merged_flat_input
