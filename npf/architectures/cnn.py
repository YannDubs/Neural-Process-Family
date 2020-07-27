import warnings

import torch
import torch.nn as nn
from torch.nn import functional as F

from npf.utils.helpers import (
    channels_to_2nd_dim,
    channels_to_last_dim,
    make_depth_sep_conv,
)
from npf.utils.initialization import init_param_, weights_init

__all__ = [
    "GaussianConv2d",
    "ConvBlock",
    "ResNormalizedConvBlock",
    "ResConvBlock",
    "CNN",
    "UnetCNN",
]


class GaussianConv2d(nn.Module):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        assert kernel_size % 2 == 1
        self.kernel_sizes = (kernel_size, kernel_size)
        self.exponent = -(
            (torch.arange(0, kernel_size).view(-1, 1).float() - kernel_size // 2) ** 2
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.weights_x = nn.Parameter(torch.tensor([1.0]))
        self.weights_y = nn.Parameter(torch.tensor([1.0]))

    def forward(self, X):
        # only switch first time to device
        self.exponent = self.exponent.to(X.device)

        marginal_x = torch.softmax(self.exponent * self.weights_x, dim=0)
        marginal_y = torch.softmax(self.exponent * self.weights_y, dim=0).T

        in_chan = X.size(1)
        filters = marginal_x @ marginal_y
        filters = filters.view(1, 1, *self.kernel_sizes).expand(
            in_chan, 1, *self.kernel_sizes
        )

        return F.conv2d(X, filters, groups=in_chan, **self.kwargs)


class ConvBlock(nn.Module):
    """Simple convolutional block with a single layer.

    Parameters
    ----------
    in_chan : int
        Number of input channels.

    out_chan : int
        Number of output channels.

    Conv : nn.Module
        Convolutional layer (unitialized). E.g. `nn.Conv1d`.

    kernel_size : int or tuple, optional
        Size of the convolving kernel.

    dilation : int or tuple, optional
        Spacing between kernel elements.

    activation: callable, optional
        Activation object. E.g. `nn.ReLU`.

    Normalization : nn.Module, optional
        Normalization layer (unitialized). E.g. `nn.BatchNorm1d`.

    kwargs :
        Additional arguments to `Conv`.

    References
    ----------
    [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016, October). Identity mappings
        in deep residual networks. In European conference on computer vision
        (pp. 630-645). Springer, Cham.

    [2] Chollet, F. (2017). Xception: Deep learning with depthwise separable
        convolutions. In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 1251-1258).
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        Conv,
        kernel_size=5,
        dilation=1,
        activation=nn.ReLU(),
        Normalization=nn.Identity,
        **kwargs
    ):
        super().__init__()
        self.activation = activation

        padding = kernel_size // 2

        Conv = make_depth_sep_conv(Conv)

        self.conv = Conv(in_chan, out_chan, kernel_size, padding=padding, **kwargs)
        self.norm = Normalization(in_chan)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X):
        return self.conv(self.activation(self.norm(X)))


class ResConvBlock(nn.Module):
    """Convolutional block inspired by the pre-activation Resnet [1]
    and depthwise separable convolutions [2].

    Parameters
    ----------
    in_chan : int
        Number of input channels.

    out_chan : int
        Number of output channels.

    Conv : nn.Module
        Convolutional layer (unitialized). E.g. `nn.Conv1d`.

    kernel_size : int or tuple, optional
        Size of the convolving kernel. Should be odd to keep the same size.

    activation: callable, optional
        Activation object. E.g. `nn.RelU()`.

    Normalization : nn.Module, optional
        Normalization layer (unitialized). E.g. `nn.BatchNorm1d`.

    n_conv_layers : int, optional
        Number of convolutional layers, can be 1 or 2.

    is_bias : bool, optional
        Whether to use a bias.

    References
    ----------
    [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016, October). Identity mappings
        in deep residual networks. In European conference on computer vision
        (pp. 630-645). Springer, Cham.

    [2] Chollet, F. (2017). Xception: Deep learning with depthwise separable
        convolutions. In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 1251-1258).
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        Conv,
        kernel_size=5,
        activation=nn.ReLU(),
        Normalization=nn.Identity,
        is_bias=True,
        n_conv_layers=1,
    ):
        super().__init__()
        self.activation = activation
        self.n_conv_layers = n_conv_layers
        assert self.n_conv_layers in [1, 2]

        if kernel_size % 2 == 0:
            raise ValueError("`kernel_size={}`, but should be odd.".format(kernel_size))

        padding = kernel_size // 2

        if self.n_conv_layers == 2:
            self.norm1 = Normalization(in_chan)
            self.conv1 = make_depth_sep_conv(Conv)(
                in_chan, in_chan, kernel_size, padding=padding, bias=is_bias
            )
        self.norm2 = Normalization(in_chan)
        self.conv2_depthwise = Conv(
            in_chan, in_chan, kernel_size, padding=padding, groups=in_chan, bias=is_bias
        )
        self.conv2_pointwise = Conv(in_chan, out_chan, 1, bias=is_bias)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X):

        if self.n_conv_layers == 2:
            out = self.conv1(self.activation(self.norm1(X)))
        else:
            out = X

        out = self.conv2_depthwise(self.activation(self.norm2(out)))
        # adds residual before point wise => output can change number of channels
        out = out + X
        out = self.conv2_pointwise(out.contiguous())  # for some reason need contiguous
        return out


class ResNormalizedConvBlock(ResConvBlock):
    """Modification of `ResConvBlock` to use normalized convolutions [1].

    Parameters
    ----------
    in_chan : int
        Number of input channels.

    out_chan : int
        Number of output channels.

    Conv : nn.Module
        Convolutional layer (unitialized). E.g. `nn.Conv1d`.

    kernel_size : int or tuple, optional
        Size of the convolving kernel. Should be odd to keep the same size.

    activation: nn.Module, optional
        Activation object. E.g. `nn.RelU()`.

    is_bias : bool, optional
        Whether to use a bias.

    References
    ----------
    [1] Knutsson, H., & Westin, C. F. (1993, June). Normalized and differential
        convolution. In Proceedings of IEEE Conference on Computer Vision and
        Pattern Recognition (pp. 515-523). IEEE.
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        Conv,
        kernel_size=5,
        activation=nn.ReLU(),
        is_bias=True,
        **kwargs
    ):
        super().__init__(
            in_chan,
            out_chan,
            Conv,
            kernel_size=kernel_size,
            activation=activation,
            is_bias=is_bias,
            Normalization=nn.Identity,
            **kwargs
        )  # make sure no normalization

    def reset_parameters(self):
        weights_init(self)
        self.bias = nn.Parameter(torch.tensor([0.0]))

        self.temperature = nn.Parameter(torch.tensor([0.0]))
        init_param_(self.temperature)

    def forward(self, X):
        """
        Apply a normalized convolution. X should contain 2*in_chan channels.
        First halves for signal, last halve for corresponding confidence channels.
        """

        signal, conf_1 = X.chunk(2, dim=1)
        # make sure confidence is in 0 1 (might not be due to the pointwise trsnf)
        conf_1 = conf_1.clamp(min=0, max=1)
        X = signal * conf_1

        numerator = self.conv1(self.activation(X))
        numerator = self.conv2_depthwise(self.activation(numerator))
        density = self.conv2_depthwise(self.conv1(conf_1))
        out = numerator / torch.clamp(density, min=1e-5)

        # adds residual before point wise => output can change number of channels

        # make sure that confidence cannot decrease and cannot be greater than 1
        conf_2 = conf_1 + torch.sigmoid(
            density * F.softplus(self.temperature) + self.bias
        )
        conf_2 = conf_2.clamp(max=1)
        out = out + X

        out = self.conv2_pointwise(out)
        conf_2 = self.conv2_pointwise(conf_2)

        return torch.cat([out, conf_2], dim=1)


class CNN(nn.Module):
    """Simple multilayer CNN.

    Parameters
    ----------
    n_channels : int or list
        Number of channels, same for input and output. If list then needs to be
        of size `n_blocks - 1`, e.g. [16, 32, 64] means that you will have a
        `[ConvBlock(16,32), ConvBlock(32, 64)]`.

    ConvBlock : nn.Module
        Convolutional block (unitialized). Needs to take as input `Should be
        initialized with `ConvBlock(in_chan, out_chan)`.

    n_blocks : int, optional
        Number of convolutional blocks.

    is_chan_last : bool, optional
        Whether the channels are on the last dimension of the input.

    kwargs :
        Additional arguments to `ConvBlock`.
    """

    def __init__(self, n_channels, ConvBlock, n_blocks=3, is_chan_last=False, **kwargs):

        super().__init__()
        self.n_blocks = n_blocks
        self.is_chan_last = is_chan_last
        self.in_out_channels = self._get_in_out_channels(n_channels, n_blocks)
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(in_chan, out_chan, **kwargs)
                for in_chan, out_chan in self.in_out_channels
            ]
        )
        self.is_return_rep = False  # never return representation for vanilla conv

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def _get_in_out_channels(self, n_channels, n_blocks):
        """Return a list of tuple of input and output channels."""
        if isinstance(n_channels, int):
            channel_list = [n_channels] * (n_blocks + 1)
        else:
            channel_list = list(n_channels)

        assert len(channel_list) == (n_blocks + 1), "{} != {}".format(
            len(channel_list), n_blocks + 1
        )

        return list(zip(channel_list, channel_list[1:]))

    def forward(self, X):
        if self.is_chan_last:
            X = channels_to_2nd_dim(X)

        X, representation = self.apply_convs(X)

        if self.is_chan_last:
            X = channels_to_last_dim(X)

        if self.is_return_rep:
            return X, representation

        return X

    def apply_convs(self, X):
        for conv_block in self.conv_blocks:
            X = conv_block(X)
        return X, None


class UnetCNN(CNN):
    """Unet [1].

    Parameters
    ----------
    n_channels : int or list
        Number of channels, same for input and output. If list then needs to be
        of size `n_blocks - 1`, e.g. [16, 32, 64] means that you will have a
        `[ConvBlock(16,32), ConvBlock(32, 64)]`.

    ConvBlock : nn.Module
        Convolutional block (unitialized). Needs to take as input `Should be
        initialized with `ConvBlock(in_chan, out_chan)`.

    Pool : nn.Module
        Pooling layer (unitialized). E.g. torch.nn.MaxPool1d.

    upsample_mode : {'nearest', 'linear', bilinear', 'bicubic', 'trilinear'}
        The upsampling algorithm: nearest, linear (1D-only), bilinear, bicubic
        (2D-only), trilinear (3D-only).

    max_nchannels : int, optional
        Bounds the maximum number of channels instead of always doubling them at
        downsampling block.

    pooling_size : int or tuple, optional
        Size of the pooling filter.

    is_force_same_bottleneck : bool, optional
        Whether to use the average bottleneck for the same functions sampled at
        different context and target. If `True` the first and second halves
        of a batch should contain different samples of the same functions (in order).

    is_return_rep : bool, optional
        Whether to return a summary representation, that corresponds to the
        bottleneck + global mean pooling.

    kwargs :
        Additional arguments to `CNN` and `ConvBlock`.

    References
    ----------
    [1] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional
        networks for biomedical image segmentation." International Conference on
        Medical image computing and computer-assisted intervention. Springer, Cham, 2015.
    """

    def __init__(
        self,
        n_channels,
        ConvBlock,
        Pool,
        upsample_mode,
        max_nchannels=256,
        pooling_size=2,
        is_force_same_bottleneck=False,
        is_return_rep=False,
        **kwargs
    ):

        self.max_nchannels = max_nchannels
        super().__init__(n_channels, ConvBlock, **kwargs)
        self.pooling_size = pooling_size
        self.pooling = Pool(self.pooling_size)
        self.upsample_mode = upsample_mode
        self.is_force_same_bottleneck = is_force_same_bottleneck
        self.is_return_rep = is_return_rep

    def apply_convs(self, X):
        n_down_blocks = self.n_blocks // 2
        residuals = [None] * n_down_blocks

        # Down
        for i in range(n_down_blocks):
            X = self.conv_blocks[i](X)
            residuals[i] = X
            X = self.pooling(X)

        # Bottleneck
        X = self.conv_blocks[n_down_blocks](X)
        # Representation before forcing same bottleneck
        representation = X.view(*X.shape[:2], -1).mean(-1)

        if self.is_force_same_bottleneck and self.training:
            # forces the u-net to use the bottleneck by giving additional information
            # there. I.e. taking average between bottleenck of different samples
            # of the same functions. Because bottleneck should be a global representation
            # => should not depend on the sample you chose
            batch_size = X.size(0)
            batch_1 = X[: batch_size // 2, ...]
            batch_2 = X[batch_size // 2 :, ...]
            X_mean = (batch_1 + batch_2) / 2
            X = torch.cat([X_mean, X_mean], dim=0)

        # Up
        for i in range(n_down_blocks + 1, self.n_blocks):
            X = F.interpolate(
                X,
                mode=self.upsample_mode,
                scale_factor=self.pooling_size,
                align_corners=True,
            )
            X = torch.cat(
                (X, residuals[n_down_blocks - i]), dim=1
            )  # concat on channels
            X = self.conv_blocks[i](X)

        return X, representation

    def _get_in_out_channels(self, n_channels, n_blocks):
        """Return a list of tuple of input and output channels for a Unet."""
        # doubles at every down layer, as in vanilla U-net
        factor_chan = 2

        assert n_blocks % 2 == 1, "n_blocks={} not odd".format(n_blocks)
        # e.g. if n_channels=16, n_blocks=5: [16, 32, 64]
        channel_list = [factor_chan ** i * n_channels for i in range(n_blocks // 2 + 1)]
        # e.g.: [16, 32, 64, 64, 32, 16]
        channel_list = channel_list + channel_list[::-1]
        # bound max number of channels by self.max_nchannels (besides first and
        # last dim as this is input / output cand sohould not be changed)
        channel_list = (
            channel_list[:1]
            + [min(c, self.max_nchannels) for c in channel_list[1:-1]]
            + channel_list[-1:]
        )
        # e.g.: [(16, 32), (32,64), (64, 64), (64, 32), (32, 16)]
        in_out_channels = super()._get_in_out_channels(channel_list, n_blocks)
        # e.g.: [(16, 32), (32,64), (64, 64), (128, 32), (64, 16)] due to concat
        idcs = slice(len(in_out_channels) // 2 + 1, len(in_out_channels))
        in_out_channels[idcs] = [
            (in_chan * 2, out_chan) for in_chan, out_chan in in_out_channels[idcs]
        ]
        return in_out_channels
