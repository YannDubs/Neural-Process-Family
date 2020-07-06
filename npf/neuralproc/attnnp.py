"""Module for attentive [conditional | latent] neural processes"""
import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.independent import Independent

from npf.architectures import (
    RelativeSinusoidalEncodings,
    SelfAttention,
    get_attender,
    merge_flat_input,
)

from .base import LatentNeuralProcessFamily, NeuralProcessFamily
from .np import CNP

__all__ = ["AttnCNP", "AttnLNP"]

logger = logging.getLogger(__name__)


class AttnCNP(NeuralProcessFamily):
    """
    Attentive conditional neural process. I.e. deterministic version of [1].

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    XYEncoder : nn.Module, optional
        Encoder module which maps {x_transf_i, y_i} -> {r_i}. C.f. ConditionalNeuralProcess for more
        details. Only used if `is_self_attn==False`.

    attention : callable or str, optional
        Type of attention to use. More details in `get_attender`.

    attention_kwargs : dict, optional
        Additional arguments to `get_attender`.

    self_attention_kwargs : dict, optional
        Additional arguments to `SelfAttention`.

    is_self_attn : bool, optional
        Whether to use self attention in the encoder. 

    kwargs :
        Additional arguments to `NeuralProcessFamily`.

    References
    ----------
    [1] Kim, Hyunjik, et al. "Attentive neural processes." arXiv preprint
        arXiv:1901.05761 (2019).
    """

    _valid_paths = ["deterministic"]

    def __init__(
        self,
        x_dim,
        y_dim,
        XYEncoder=None,
        attention="scaledot",
        attention_kwargs={},
        self_attention_kwargs={},
        is_self_attn=False,
        **kwargs,
    ):

        # don't force det so that can inherit ,
        kwargs["encoded_path"] = kwargs.get("encoded_path", "deterministic")
        super().__init__(
            x_dim, y_dim, **kwargs,
        )

        self.is_self_attn = is_self_attn

        if self.is_self_attn:
            XYEncoder = merge_flat_input(
                SelfAttention, is_sum_merge=True, **self_attention_kwargs
            )
        elif XYEncoder is None:
            XYEncoder = self.dflt_Modules["XYEncoder"]

        self.xy_encoder = XYEncoder(self.x_transf_dim, self.y_dim, self.r_dim)

        self.attender = get_attender(
            attention, self.x_transf_dim, self.r_dim, self.r_dim, **attention_kwargs
        )

        self.reset_parameters()

    dflt_Modules = CNP.dflt_Modules

    def encode_globally(self, X_cntxt, Y_cntxt):
        batch_size, n_cntxt, _ = X_cntxt.shape

        if n_cntxt == 0:
            # arbitrarily setting each target representation to zero when no context
            R_cntxt = torch.zeros(batch_size, 0, self.r_dim, device=X_cntxt.device)
        else:
            # One representation per context point
            # size = [batch_size, n_cntxt, r_dim]
            R_cntxt = self.xy_encoder(X_cntxt, Y_cntxt)

        return R_cntxt

    def trgt_dependent_representation(self, X_cntxt, _, R, X_trgt):
        batch_size, n_cntxt, _ = X_cntxt.shape

        if n_cntxt == 0:
            # arbitrarily setting each target representation to zero when no context
            R_trgt = torch.zeros(
                batch_size, X_trgt.size(1), self.r_dim, device=R.device
            )
        else:
            # size = [batch_size, n_trgt, r_dim]
            R_trgt = self.attender(X_cntxt, X_trgt, R)  # keys, queries, values

        # n_z_samples=1. size = [1, batch_size, n_trgt, r_dim]
        return R_trgt.unsqueeze(0)


class AttnLNP(LatentNeuralProcessFamily, AttnCNP):
    """
    Attentive (latent) neural process [1].

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    LatentEncoder : nn.Module, optional
        Encoder which maps r -> z_suffstat. It should be constructed via
        `LatentEncoder(r_dim, n_out)`.  If `None` uses an MLP.

    kwargs : 
        Additional arguments to `AttnCNP` and `NeuralProcessFamily`.

    References
    ----------
    [1] Kim, Hyunjik, et al. "Attentive neural processes." arXiv preprint
        arXiv:1901.05761 (2019).
    """

    _valid_paths = ["both"]

    def __init__(self, x_dim, y_dim, **kwargs):
        super().__init__(x_dim, y_dim, encoded_path="both", **kwargs)

    @property
    def dflt_Modules(self):
        dflt_Modules = AttnCNP.dflt_Modules.__get__(self)
        dflt_Modules2 = LatentNeuralProcessFamily.dflt_Modules.__get__(self)
        dflt_Modules.update(dflt_Modules2)

        return dflt_Modules

    def rep_to_lat_input(self, R):
        batch_size, n_cntxt, _ = R.shape

        if n_cntxt == 0:
            # arbitrarily setting the global representation for latent path to zero when no context
            R = torch.zeros(batch_size, 1, self.r_dim, device=R.device)

        # one deterministic representation for each context. But need single latent => pool
        # size = [batch_size, 1, r_dim]
        return torch.mean(R, dim=1, keepdim=True)

    def trgt_dependent_representation(self, X_cntxt, z_samples, R, X_trgt):

        batch_size, n_trgt, _ = X_trgt.shape
        n_z_samples = z_samples.size(0)

        # latent path
        # size = [n_z_samples, batch_size, n_trgt, r_dim]
        z_samples = z_samples.expand(n_z_samples, batch_size, n_trgt, self.r_dim)

        # deterministic path
        # size = [batch_size, n_trgt, r_dim]
        R_trgt_det = AttnCNP.trgt_dependent_representation(
            self, X_cntxt, _, R, X_trgt
        ).squeeze(0)

        # merging both
        # size = [n_z_samples, batch_size, n_trgt, r_dim]
        R_trgt = self.merge_r_z(R_trgt_det, z_samples)

        return R_trgt
