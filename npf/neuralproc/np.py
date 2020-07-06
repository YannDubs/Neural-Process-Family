"""Module for vanilla  [conditional | latent] neural processes"""
import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal

from npf.architectures import MLP, merge_flat_input

from .base import LatentNeuralProcessFamily, NeuralProcessFamily

logger = logging.getLogger(__name__)

__all__ = ["CNP", "LNP"]


class CNP(NeuralProcessFamily):
    """
    Conditional Neural Process from [1].

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    XYEncoder : nn.Module, optional
        Encoder module which maps {x_transf_i, y_i} -> {r_i}. It should be constructable
        via `XYEncoder(x_transf_dim, y_dim, n_out)`. If you have an encoder that maps
        [x;y] -> r you can convert it via `merge_flat_input(Encoder)`. `None` uses
        MLP. In the computational model this corresponds to `h` (with XEncoder). 
        Example:
            - `merge_flat_input(MLP, is_sum_merge=False)` : learn representation
            with MLP. `merge_flat_input` concatenates (or sums) X and Y inputs.
            - `merge_flat_input(SelfAttention, is_sum_merge=True)` : self attention mechanisms as 
            [4]. For more parameters (attention type, number of layers ...) refer to its docstrings.
            - `discard_ith_arg(MLP, 0)` if want the encoding to only depend on Y.

    kwargs : 
        Additional arguments to `NeuralProcessFamily`

    References
    ----------
    [1] Garnelo, Marta, et al. "Conditional neural processes." arXiv preprint
        arXiv:1807.01613 (2018).
    """

    _valid_paths = ["deterministic"]

    def __init__(self, x_dim, y_dim, XYEncoder=None, **kwargs):

        # don't force det so that can inherit ,
        kwargs["encoded_path"] = kwargs.get("encoded_path", "deterministic")
        super().__init__(
            x_dim, y_dim, **kwargs,
        )

        if XYEncoder is None:
            XYEncoder = self.dflt_Modules["XYEncoder"]

        self.xy_encoder = XYEncoder(self.x_transf_dim, self.y_dim, self.r_dim)

        self.reset_parameters()

    @property
    def dflt_Modules(self):
        # allow inheritence
        dflt_Modules = NeuralProcessFamily.dflt_Modules.__get__(self)

        SubXYEncoder = partial(
            MLP, n_hidden_layers=2, is_force_hid_smaller=True, hidden_size=self.r_dim,
        )
        dflt_Modules["XYEncoder"] = merge_flat_input(SubXYEncoder, is_sum_merge=True)

        return dflt_Modules

    def encode_globally(self, X_cntxt, Y_cntxt):
        batch_size, n_cntxt, _ = X_cntxt.shape

        # encode all cntxt pair separately
        # size = [batch_size, n_cntxt, r_dim]
        R_cntxt = self.xy_encoder(X_cntxt, Y_cntxt)

        # using mean for aggregation (i.e. n_rep=1)
        # size = [batch_size, 1, r_dim]
        R = torch.mean(R_cntxt, dim=1, keepdim=True)

        if n_cntxt == 0:
            # arbitrarily setting the global representation to zero when no context
            R = torch.zeros(batch_size, 1, self.r_dim, device=R_cntxt.device)

        return R

    def trgt_dependent_representation(self, _, __, R, X_trgt):

        # same (global) representation for predicting all target point
        batch_size, n_trgt, _ = X_trgt.shape
        R_trgt = R.expand(batch_size, n_trgt, self.r_dim)

        # n_z_samples=1. size = [1, batch_size, n_trgt, r_dim]
        return R_trgt.unsqueeze(0)


class LNP(LatentNeuralProcessFamily, CNP):
    """
    (Latent) Neural process from [1].

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    encoded_path : {"latent", "both"}
        Which path(s) to use:
        - `"latent"`  the decoder gets a sample latent representation as input as in [1].
        - `"both"` concatenates both the deterministic and sampled latents as input to the decoder [2].

    kwargs : 
        Additional arguments to `ConditionalNeuralProcess` and `NeuralProcessFamily`.

    References
    ----------
    [1] Garnelo, Marta, et al. "Neural processes." arXiv preprint
        arXiv:1807.01622 (2018).
    [2] Kim, Hyunjik, et al. "Attentive neural processes." arXiv preprint
        arXiv:1901.05761 (2019).
    """

    def __init__(self, x_dim, y_dim, encoded_path="latent", **kwargs):
        super().__init__(x_dim, y_dim, encoded_path=encoded_path, **kwargs)

    def trgt_dependent_representation(self, X_cntxt, z_samples, R, X_trgt):

        batch_size, n_trgt, _ = X_trgt.shape
        n_z_samples = z_samples.size(0)

        if self.encoded_path == "both":
            # size = [n_z_samples, batch_size, 1, r_dim]
            R_trgt = self.merge_r_z(R, z_samples)

        elif self.encoded_path == "latent":
            # size = [n_z_samples, batch_size, 1, r_dim]
            R_trgt = z_samples

        R_trgt = R_trgt.expand(n_z_samples, batch_size, n_trgt, self.r_dim)

        return R_trgt
