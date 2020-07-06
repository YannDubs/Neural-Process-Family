import torch.nn as nn

from npf.utils.initialization import weights_init
from .attention import get_attender
from .encoders import SinusoidalEncodings, RelativeSinusoidalEncodings

__all__ = ["SelfAttention"]


class SelfAttention(nn.Module):
    """Self Attention Layer.

    Parameters
    ----------
    x_dim : int
        Input dimension.

    out_dim : int
        Output dimension. If not None will do all the computation
        with a size of `x_dim` and add a linear layer at the end to reshape.

    n_attn_layers : int, optional
        Number of self attention layers.

    attention : callable or str, optional
        Type of attention to use. More details in `get_attender`.

    positional : {"absolute", "relative", None}, optional
        Type of positional encodings. `"absolute"` adds positional encodings
        (sinusoidals) to the input before self attention (Transformer). `"relative"`
        uses relative encodings at every attention layer (Transformer XL).
        `position_dim` has to be given when not `None`.

    position_dim : int, optional
        Dimenion of the position.

    max_len : int, optional
        Maximum number of x. Only used if `positional is not None`.

    kwargs :
        Additional arguments to `get_attender`.
    """

    def __init__(
        self,
        x_dim,
        out_dim=None,
        n_attn_layers=2,
        attention="transformer",
        positional=None,
        position_dim=None,
        max_len=2000,
        **kwargs
    ):
        super().__init__()
        self.positional = positional

        if self.positional == "absolute":
            self.pos_encoder = SinusoidalEncodings(position_dim, x_dim)
        elif self.positional == "relative":
            self.rel_pos_encoder = RelativeSinusoidalEncodings(position_dim, x_dim)
            kwargs["is_relative_pos"] = True
        elif self.positional is None:
            pass
        else:
            raise ValueError("Unknown positional={}.".format(positional))

        self.attn_layers = nn.ModuleList(
            [
                get_attender(attention, x_dim, x_dim, x_dim, **kwargs)
                for _ in range(n_attn_layers)
            ]
        )

        self.is_resize = out_dim is not None
        if self.is_resize:
            self.resize = nn.Linear(x_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X, positions=None):
        add_to_keys = 0
        if self.positional == "absolute":
            X = X + self.pos_encoder(positions)
        elif self.positional == "relative":
            # n^2 for now but could be n(n+1)/2 if sorted
            add_to_keys = self.rel_pos_encoder(positions, positions)

        out = X

        for attn_layer in self.attn_layers:
            out = attn_layer(out + add_to_keys, out, out)

        if self.is_resize:
            out = self.resize(out)

        return out
