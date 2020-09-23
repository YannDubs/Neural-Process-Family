"""Module for convolutional [conditional | latent] neural processes"""
import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.independent import Independent

from npf.architectures import CNN, ResConvBlock, SetConv, discard_ith_arg
from npf.utils.initialization import weights_init

from .base import LatentNeuralProcessFamily, NeuralProcessFamily
from .helpers import (
    collapse_z_samples_batch,
    pool_and_replicate_middle,
    replicate_z_samples,
)
from .np import LNP

logger = logging.getLogger(__name__)

__all__ = ["ConvCNP", "ConvLNP"]


class ConvCNP(NeuralProcessFamily):
    """
    Convolutional conditional neural process [1].

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    density_induced : int, optional
        Density of induced-inputs to use. The induced-inputs will be regularly sampled.

    Interpolator : callable or str, optional
        Callable to use to compute cntxt / trgt to and from the induced points.  {(x^k, y^k)}, {x^q} -> {y^q}.
        It should be constructed via `Interpolator(x_dim, in_dim, out_dim)`. Example:
            - `SetConv` : uses a set convolution as in the paper.
            - `"TransformerAttender"` : uses a cross attention layer.

    CNN : nn.Module, optional
        Convolutional model to use between induced points. It should be constructed via
        `CNN(r_dim)`. Important : the channel needs to be last dimension of input. Example:
            - `partial(CNN,ConvBlock=ResConvBlock,Conv=nn.Conv2d,is_chan_last=True` : uses a small
            ResNet.
            - `partial(UnetCNN,ConvBlock=ResConvBlock,Conv=nn.Conv2d,is_chan_last=True` : uses a
            UNet.

    kwargs :
        Additional arguments to `NeuralProcessFamily`.

    References
    ----------
    [1] Gordon, Jonathan, et al. "Convolutional conditional neural processes." arXiv preprint
    arXiv:1910.13556 (2019).
    """

    _valid_paths = ["deterministic"]

    def __init__(
        self,
        x_dim,
        y_dim,
        density_induced=128,
        Interpolator=SetConv,
        CNN=partial(
            CNN,
            ConvBlock=ResConvBlock,
            Conv=nn.Conv1d,
            n_blocks=3,
            Normalization=nn.Identity,
            is_chan_last=True,
            kernel_size=11,
        ),
        **kwargs,
    ):

        if (
            "Decoder" in kwargs and kwargs["Decoder"] != nn.Identity
        ):  # identity means that not using
            logger.warning(
                "`Decoder` was given to `ConvCNP`. To be translation equivariant you should disregard the first argument for example using `discard_ith_arg(Decoder, i=0)`, which is done by default when you DO NOT provide the Decoder."
            )

        # don't force det so that can inherit ,
        kwargs["encoded_path"] = kwargs.get("encoded_path", "deterministic")
        super().__init__(
            x_dim,
            y_dim,
            x_transf_dim=None,
            XEncoder=nn.Identity,
            **kwargs,
        )

        self.density_induced = density_induced
        # input is between -1 and 1 but use at least 0.5 temporary values on each sides to not
        # have strong boundary effects
        self.X_induced = torch.linspace(-1.5, 1.5, int(self.density_induced * 3))
        self.CNN = CNN

        self.cntxt_to_induced = Interpolator(self.x_dim, self.y_dim, self.r_dim)
        self.induced_to_induced = CNN(self.r_dim)
        self.induced_to_trgt = Interpolator(self.x_dim, self.r_dim, self.r_dim)

        self.reset_parameters()

    @property
    def n_induced(self):
        # using property because this might change after you set extrapolation
        return len(self.X_induced)

    @property
    def dflt_Modules(self):
        # allow inheritence
        dflt_Modules = NeuralProcessFamily.dflt_Modules.__get__(self)

        # don't depend on x
        dflt_Modules["Decoder"] = discard_ith_arg(dflt_Modules["SubDecoder"], i=0)

        return dflt_Modules

    def _get_X_induced(self, X):
        batch_size, _, _ = X.shape

        # effectively puts on cuda only once
        self.X_induced = self.X_induced.to(X.device)
        X_induced = self.X_induced.view(1, -1, 1)
        X_induced = X_induced.expand(batch_size, self.n_induced, self.x_dim)
        return X_induced

    def encode_globally(self, X_cntxt, Y_cntxt):
        batch_size, n_cntxt, _ = X_cntxt.shape

        # size = [batch_size, n_induced, x_dim]
        X_induced = self._get_X_induced(X_cntxt)

        # size = [batch_size, n_induced, r_dim]
        R_induced = self.cntxt_to_induced(X_cntxt, X_induced, Y_cntxt)

        if n_cntxt == 0:
            # arbitrarily setting the global representation to zero when no context
            # but the density channel will also be => makes sense
            R_induced = torch.zeros(
                batch_size, self.n_induced, self.r_dim, device=R_induced.device
            )

        # size = [batch_size, n_induced, r_dim]
        R_induced = self.induced_to_induced(R_induced)

        return R_induced

    def trgt_dependent_representation(self, X_cntxt, z_samples, R_induced, X_trgt):
        batch_size, n_trgt, _ = X_trgt.shape

        # size = [batch_size, n_induced, x_dim]
        X_induced = self._get_X_induced(X_cntxt)

        # size = [batch_size, n_trgt, r_dim]
        R_trgt = self.induced_to_trgt(X_induced, X_trgt, R_induced)

        # n_z_samples=1. size = [1, batch_size, n_trgt, r_dim]
        return R_trgt.unsqueeze(0)

    def set_extrapolation(self, min_max):
        """
        Scale the induced inputs to be in a given range while keeping
        the same density than during training (used for extrapolation.).
        """
        current_min = min_max[0] - 0.5
        current_max = min_max[1] + 0.5
        self.X_induced = torch.linspace(
            current_min,
            current_max,
            int(self.density_induced * (current_max - current_min)),
        )


class ConvLNP(LatentNeuralProcessFamily, ConvCNP):
    """
    Convolutional latent neural process [1].

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    is_global : bool, optional
        Whether to also use a global representation in addition to the latent one. Only if
        encoded_path = `latent`.

    CNNPostZ : Module, optional
        CNN to use after the sampling. If `None` uses the same as before sampling. Note that computations
        will be heavier after sampling (as performing on all the samples) so you might want to
        make it smaller.

    kwargs :
        Additional arguments to `ConvCNP`.

    References
    ----------
    [1] Foong, Andrew YK, et al. "Meta-Learning Stationary Stochastic Process Prediction with
    Convolutional Neural Processes." arXiv preprint arXiv:2007.01332 (2020).
    """

    _valid_paths = ["latent", "both"]

    def __init__(
        self,
        x_dim,
        y_dim,
        CNNPostZ=None,
        encoded_path="latent",
        is_global=False,
        **kwargs,
    ):
        super().__init__(
            x_dim,
            y_dim,
            encoded_path=encoded_path,
            **kwargs,
        )

        self.is_global = is_global

        if CNNPostZ is None:
            CNNPostZ = self.CNN

        self.induced_to_induced_post_sampling = CNNPostZ(self.r_dim)

        self.reset_parameters()

    @property
    def dflt_Modules(self):
        # allow inheritance
        dflt_Modules = ConvCNP.dflt_Modules.__get__(self)
        dflt_Modules2 = LatentNeuralProcessFamily.dflt_Modules.__get__(self)
        dflt_Modules.update(dflt_Modules2)

        # use smaller decoder than ConvCNP because more param due to `induced_to_induced_post_sampling`
        dflt_Modules["Decoder"] = discard_ith_arg(nn.Linear, i=0)

        return dflt_Modules

    def rep_to_lat_input(self, R):
        batch_size, *n_induced, _ = R.shape

        if self.encoded_path == "latent":
            # size = [batch_size, *n_induced, r_dim]
            return R

        elif self.encoded_path == "both":
            # size = [batch_size, 1, r_dim]
            return R.view(batch_size, -1, self.r_dim).mean(dim=1, keepdim=True)

    def trgt_dependent_representation(self, X_cntxt, z_samples, R_induced, X_trgt):

        batch_size, n_trgt, _ = X_trgt.shape
        n_z_samples, _, n_lat, __ = z_samples.shape

        # size = [batch_size, n_induced, x_dim]
        X_induced = self._get_X_induced(X_cntxt)

        # size = [n_z_samples*batch_size, *, x_dim]
        X_induced = collapse_z_samples_batch(
            replicate_z_samples(X_induced, n_z_samples)
        )
        X_trgt = collapse_z_samples_batch(replicate_z_samples(X_trgt, n_z_samples))

        if self.encoded_path == "latent":
            # make all computations with n_z_samples and batch size merged (because CNN need it)
            # size = [n_z_samples * batch_size, n_induced, z_dim]
            z_samples = collapse_z_samples_batch(z_samples)

            # size = [n_z_samples * batch_size, n_induced, r_dim]
            if self.z_dim != self.r_dim:
                z_samples = self.reshaper_z(z_samples)

            # size = [n_z_samples*batch_size, n_induced, r_dim]
            # "mixing" after the sampling to have coherent samples
            z_samples = self.induced_to_induced_post_sampling(z_samples)

            #! SHOULD be directly after sampling (like in gridconvnp)
            if self.is_global:
                # size = [n_z_samples*batch_size, n_induced, r_dim]
                z_samples = self.add_global_latent(z_samples)

            # size = [n_z_samples * batch_size, n_trgt, r_dim]
            R_trgt = self.induced_to_trgt(X_induced, X_trgt, z_samples)

        elif self.encoded_path == "both":
            # size = [n_z_samples, batch_size, n_induced, z_dim]
            z_samples = z_samples.expand(
                n_z_samples, batch_size, self.n_induced, self.z_dim
            )

            R_induced = self.merge_r_z(R_induced, z_samples)

            # make all computations with n_z_samples and batch size merged (because CNN need it)
            # size = [n_z_samples * batch_size, self.n_induced, self.r_dim]
            R_induced = collapse_z_samples_batch(R_induced)

            # to make it comparable with `latent` path
            R_induced = self.induced_to_induced_post_sampling(R_induced)

            # size = [n_z_samples * batch_size, n_trgt, r_dim]
            R_trgt = self.induced_to_trgt(X_induced, X_trgt, R_induced)

        # extracts n_z_dim
        R_trgt = R_trgt.view(n_z_samples, batch_size, n_trgt, self.r_dim)

        return R_trgt

    def add_global_latent(self, z_samples):
        """Add a global latent to z_samples."""
        # size = [n_z_samples*batch_size, n_induced, r_dim // 2]
        local_z_samples, global_z_samples = z_samples.split(
            z_samples.shape[-1] // 2, dim=-1
        )

        # size = [n_z_samples*batch_size, n_induced, r_dim //2]
        global_z_samples = pool_and_replicate_middle(global_z_samples)

        # size = [n_z_samples*batch_size, n_induced * 2, r_dim]
        z_samples = torch.cat([local_z_samples, global_z_samples], dim=-1)

        return z_samples
