"""Module for on the grid convolutional [conditional | latent] neural processes"""

import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from npf.architectures import CNN, ResConvBlock
from npf.utils.helpers import (
    channels_to_2nd_dim,
    channels_to_last_dim,
    make_abs_conv,
    prod,
)

from .base import LatentNeuralProcessFamily, NeuralProcessFamily
from .convnp import ConvCNP, ConvLNP
from .helpers import collapse_z_samples_batch

__all__ = ["GridConvCNP", "GridConvLNP"]

logger = logging.getLogger(__name__)


class GridConvCNP(NeuralProcessFamily):
    """
    Spacial case of Convolutional Conditional Neural Process [1] when the context, targets and
    induced points points are on a grid of the same size.

    Notes
    -----
    - Assumes that input, output and induced points are on the same grid. I.e. This cannot be used 
    for sub-pixel interpolation / super resolution. I.e. in the code *n_rep = *n_cntxt = *n_trgt =* grid_shape.
    The real number of ontext and target will be determined by the masks.
    - Assumes that Y_cntxt is the grid values (y_dim / channels on last dim),
    while X_cntxt and X_trgt are confidence masks of the shape of the grid rather
    than set of features.
    - As X_cntxt and X_trgt is a grid, each batch example could have a different number of
    contexts  and targets (i.e. different number of non zeros).
    - As we do not use a set convolution, the receptive field is easy to specify,
    making the model much more computationally efficient.

    Parameters
    ----------
    x_dim : int
        Dimension of features. As the features are now masks, this has to be either 1 or y_dim
        as they will be multiplied to Y (with possible broadcasting). If 1 then selectign all channels
        or none.

    y_dim : int
        Dimension of y values.

    Conv : nn.Module, optional
        Convolution layer to use to map from context to induced points {(x^k, y^k)}, {x^q} -> {y^q}.

    CNN : nn.Module, optional
        Convolutional model to use between induced points. It should be constructed via 
        `CNN(r_dim)`. Important : the channel needs to be last dimension of input. Example:
            - `partial(CNN,ConvBlock=ResConvBlock,Conv=nn.Conv2d,is_chan_last=True` : uses a small 
            ResNet.
            - `partial(UnetCNN,ConvBlock=ResConvBlock,Conv=nn.Conv2d,is_chan_last=True` : uses a 
            UNet.

    kwargs :
        Additional arguments to `ConvCNP`.

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
        # uses only depth wise + make sure positive to be interpreted as a density
        Conv=lambda y_dim: make_abs_conv(nn.Conv2d)(
            y_dim, y_dim, groups=y_dim, kernel_size=11, padding=11 // 2, bias=False,
        ),
        CNN=partial(
            CNN,
            ConvBlock=ResConvBlock,
            Conv=nn.Conv2d,
            n_blocks=3,
            Normalization=nn.Identity,
            is_chan_last=True,
            kernel_size=11,
        ),
        **kwargs,
    ):

        assert (
            x_dim == 1 or x_dim == y_dim
        ), "Ensure that featrue masks can be multiplied with Y"

        if (
            "Decoder" in kwargs and kwargs["Decoder"] != nn.Identity
        ):  # identity means that not using
            logger.warning(
                "`Decoder` was given to `ConvCNP`. To be translation equivariant you should disregard the first argument for example using `discard_ith_arg(Decoder, i=0)`, which is done by default when you DO NOT provide the Decoder."
            )

        # don't force det so that can inherit ,
        kwargs["encoded_path"] = kwargs.get("encoded_path", "deterministic")
        super().__init__(
            x_dim, y_dim, x_transf_dim=None, XEncoder=nn.Identity, **kwargs,
        )

        self.CNN = CNN
        self.conv = Conv(y_dim)
        self.resizer = nn.Linear(
            self.y_dim * 2, self.r_dim
        )  # 2 because also confidence channels

        self.induced_to_induced = CNN(self.r_dim)

        self.reset_parameters()

    dflt_Modules = ConvCNP.dflt_Modules

    def cntxt_to_induced(self, mask_cntxt, X):
        """Infer the missing values  and compute a density channel."""

        # channels have to be in second dimension for convolution
        # size = [batch_size, y_dim, *grid_shape]
        X = channels_to_2nd_dim(X)
        # size = [batch_size, x_dim, *grid_shape]
        mask_cntxt = channels_to_2nd_dim(mask_cntxt).float()

        # size = [batch_size, y_dim, *grid_shape]
        X_cntxt = X * mask_cntxt
        signal = self.conv(X_cntxt)
        density = self.conv(mask_cntxt.expand_as(X))

        # normalize
        out = signal / torch.clamp(density, min=1e-5)

        # size = [batch_size, y_dim * 2, *grid_shape]
        out = torch.cat([out, density], dim=1)

        # size = [batch_size, *grid_shape, y_dim * 2]
        out = channels_to_last_dim(out)

        # size = [batch_size, *grid_shape, r_dim]
        out = self.resizer(out)

        return out

    def encode_globally(self, mask_cntxt, X):

        # size = [batch_size, *grid_shape, r_dim]
        R_induced = self.cntxt_to_induced(mask_cntxt, X)
        R_induced = self.induced_to_induced(R_induced)

        return R_induced

    def trgt_dependent_representation(self, _, __, R_induced, ___):

        # n_z_samples=1. size = [1, batch_size, n_trgt, r_dim]
        return R_induced.unsqueeze(0)

    def set_extrapolation(self, min_max):
        raise NotImplementedError("GridConvCNP cannot be used for extrapolation.")


class GridConvLNP(LatentNeuralProcessFamily, GridConvCNP):
    """
    Spacial case of Convolutional Latent Neural Process [1] when the context, targets and
    induced points points are on a grid of the same size. C.f. `GridConvCNP` for more details.

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
    [1] Gordon, Jonathan, et al. "Convolutional conditional neural processes." arXiv preprint 
    arXiv:1910.13556 (2019).
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
            x_dim, y_dim, encoded_path=encoded_path, **kwargs,
        )

        self.is_global = is_global

        if CNNPostZ is None:
            CNNPostZ = self.CNN

        self.induced_to_induced_post_sampling = CNNPostZ(self.r_dim)

        self.reset_parameters()

    # maybe should inherit from ConvLNP instead ?
    dflt_Modules = ConvLNP.dflt_Modules
    add_global_latent = ConvLNP.add_global_latent
    rep_to_lat_input = ConvLNP.rep_to_lat_input

    def trgt_dependent_representation(self, X_cntxt, z_samples, R_induced, X_trgt):
        batch_size, *grid_shape, _ = X_trgt.shape
        n_z_samples = z_samples.size(0)

        if self.encoded_path == "latent":

            # make all computations with n_z_samples and batch size merged (because CNN need it)
            # size = [n_z_samples*batch_size, *grid_shape, r_dim]
            z_samples = collapse_z_samples_batch(z_samples)

            if self.is_global:
                # size = [n_z_samples*batch_size, *grid_shape, r_dim]
                z_samples = self.add_global_latent(z_samples)

            # size = [n_z_samples*batch_size, *grid_shape, r_dim]
            # "mixing" after the sampling to have coherent samples
            R_trgt = self.induced_to_induced_post_sampling(z_samples)

        elif self.encoded_path == "both":
            # z_samples is size = [n_z_samples, batch_size, 1, r_dim]
            z_samples = z_samples.view(
                n_z_samples, batch_size, *([1] * len(grid_shape)), self.r_dim
            )
            z_samples = z_samples.expand(
                n_z_samples, batch_size, *grid_shape, self.r_dim
            )

            # size = [n_z_samples, batch_size, *grid_shape, r_dim]
            R_induced = self.merge_r_z(R_induced, z_samples)

            # make all computations with n_z_samples and batch size merged (because CNN need it)
            # size = [n_z_samples*batch_size, *grid_shape, r_dim]
            R_induced = collapse_z_samples_batch(R_induced)

            # size = [n_z_samples*batch_size, *grid_shape, r_dim]
            # to make it comparable with `latent` path
            R_trgt = self.induced_to_induced_post_sampling(R_induced)

        # extracts n_z_dim
        R_trgt = R_trgt.view(n_z_samples, batch_size, *grid_shape, self.r_dim)

        return R_trgt


"""
class GridConvLNPDet(GridConvCNP):
    _valid_paths = ["deterministic"]

    def __init__(self, *args, **kwargs):
        GridConvLNP.__init__(self, *args, encoded_path="deterministic", **kwargs)

    dflt_Modules = ConvLNP.dflt_Modules

    def trgt_dependent_representation(self, _, __, R_induced, ___):
        # size = [batch_size, n_trgt, r_dim]
        R_trgt = self.induced_to_induced_post_sampling(R_induced)

        # n_z_samples=1. size = [1, batch_size, n_trgt, r_dim]
        return R_trgt.unsqueeze(0)
"""
