import contextlib
import io
import warnings

import imageio
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cbook import MatplotlibDeprecationWarning

__all__ = ["giffify", "plot_config"]


def fig2img(fig, dpi=200):
    """Convert a Matplotlib figure to a imageio Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img = imageio.imread(buf)
    return img


@contextlib.contextmanager
def plot_config(
    style="ticks",
    context="notebook",
    palette="deep",
    font_scale=1,
    font="sans-serif",
    rc=dict(),
    set_kwargs=dict(),
    despine_kwargs=dict(),
):
    """Temporary seaborn and matplotlib figure style / context / limits / ....

    Parameters
    ----------
    style : dict, None, or one of {darkgrid, whitegrid, dark, white, ticks}
        A dictionary of parameters or the name of a preconfigured set.

    context : dict, None, or one of {paper, notebook, talk, poster}
        A dictionary of parameters or the name of a preconfigured set.

    palette : string or sequence
        Color palette, see :func:`color_palette`

    font : string
        Font family, see matplotlib font manager.

    font_scale : float, optional
        Separate scaling factor to independently scale the size of the
        font elements.

    rc : dict, optional
        Parameter mappings to override the values in the preset seaborn
        style dictionaries.

    set_kwargs : dict, optional
        kwargs for matplotlib axes. Such as xlim, ylim, ...

    despine_kwargs : dict, optional
        Arguments to `sns.despine`.
    """
    defaults = plt.rcParams.copy()

    try:
        rc["font.family"] = font
        plt.rcParams.update(rc)

        with sns.axes_style(style=style, rc=rc), sns.plotting_context(
            context=context, font_scale=font_scale, rc=rc
        ), sns.color_palette(palette):
            yield
            last_fig = plt.gcf()
            for i, ax in enumerate(last_fig.axes):
                ax.set(**set_kwargs)

            sns.despine(**despine_kwargs)

    finally:
        with warnings.catch_warnings():
            # filter out depreciation warnings when resetting defaults
            warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
            # reset defaults
            plt.rcParams.update(defaults)


def giffify(
    save_filename,
    gen_single_fig,
    sweep_parameter,
    sweep_values,
    fps=2,
    dpi=200,
    **kwargs
):
    """Make a gif by calling `single_fig` with varying parameters.
    
    Parameters
    ----------
    save_filename : str
        name fo the file for saving the gif.
        
    gen_single_fig : callable
        Function which returns a matplotlib figure. 
        
    sweep_parameter : str
        Name of the parameter to `single_fig` that will be swept over.
        
    sweep_values : array-like
        Values to sweep over.
        
    fps : int, optional
        Number of frame per second. I.e. speed of gif.
        
    dpi : int, optional
        The resolution in dots per inch.
        
    kwargs : 
        Arguments to `single_fig` that should not be swept over.
    """
    figs = []
    for v in sweep_values:
        fig = gen_single_fig(**{sweep_parameter: v}, **kwargs)
        plt.close()
        img = fig2img(fig, dpi=dpi)
        figs.append(img)

    imageio.mimsave(save_filename, figs, fps=fps)
