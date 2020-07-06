import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch

from npf.neuralproc.base import LatentNeuralProcessFamily
from npf.utils.helpers import rescale_range
from utils.helpers import set_seed

from .helpers import plot_config

DFLT_FIGSIZE = (11, 5)

__all__ = [
    "plot_dataset_samples_1d",
    "plot_prior_samples_1d",
    "plot_posterior_samples_1d",
    "plot_losses",
]


def plot_losses(
    history, title=None, figsize=DFLT_FIGSIZE, ax=None, mode="both", label_sfx=""
):
    """Plot the `training`, `validation` or `both` losses given a skorch history."""

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if mode in ["both", "validation"]:
        valid_losses = [ep["valid_loss"] for ep in history]
        ax.plot(valid_losses, label="Validation" + label_sfx)

    if mode in ["both", "training"]:
        train_losses = [ep["train_loss"] for ep in history]
        ax.plot(train_losses, label="Training" + label_sfx)

    ax.legend()
    ax.set_ylabel("Negative Log Likelihood")
    ax.set_xlabel("Number of Epochs")

    if title is not None:
        ax.set_title(title)

    return ax


def plot_dataset_samples_1d(
    dataset,
    n_samples=10,
    title="Dataset",
    figsize=DFLT_FIGSIZE,
    ax=None,
    plot_config_kwargs={},
    seed=123,
):
    """Plot `n_samples` samples of the a datset."""
    np.random.seed(seed)

    with plot_config(plot_config_kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        alpha = 0.5 + 1 / (n_samples ** 0.5 + 1)

        for i in range(n_samples):
            x, y = dataset[np.random.randint(len(dataset))]
            x = rescale_range(x, (-1, 1), dataset.min_max)
            ax.plot(x.numpy(), y.numpy(), alpha=alpha)
            ax.set_xlim(*dataset.min_max)

        if title is not None:
            ax.set_title(title, fontsize=14)

        return ax


def plot_prior_samples_1d(
    model, test_min_max=None, train_min_max=(-2, 2), n_trgt=256, **kwargs
):
    """
    Plot the mean at `n_trgt` different points for `n_samples`
    different latents (i.e. sampled functions).
    """
    if test_min_max is None:
        test_min_max = train_min_max

    input_min_max = tuple(rescale_range(np.array(test_min_max), train_min_max, (-1, 1)))
    X_trgt = torch.Tensor(np.linspace(*input_min_max, n_trgt))
    X_trgt = X_trgt.view(1, -1, 1)

    ax = _plot_posterior_predefined_cntxt(model, None, None, X_trgt, **kwargs)

    return ax


def plot_posterior_samples_1d(
    X,
    Y,
    get_cntxt_trgt,
    model,
    compare_model=None,
    model_labels=dict(main="Model", compare="Compare", generator="Oracle GP"),
    generator=None,
    is_plot_real=True,
    train_min_max=(-2, 2),
    ax=None,
    seed=None,
    is_fill_generator_std=True,
    y_lim=(None, None),
    plot_config_kwargs={},
    **kwargs,
):
    """
    Plot and compare (samples from) the conditional posterior predictive estimated
    by some models at random cntxt and target points.

    Parameters
    ----------
    X : torch.tensor, size=[1, n_trgt, x_dim]
        All features X, should be rescaled shuch that interpolation is in (-1,1).

    Y : torch.tensor, size=[1, n_trgt, x_dim]
        Actual values Y for all X.

    get_cntxt_trgt : callable
        Function that takes as input the features and tagrets `X`, `y` and return
        the corresponding `X_cntxt, Y_cntxt, X_trgt, Y_trgt`.

    model : nn.Module
        Main prediction model.

    compare_model : nn.Module, optional
        Secondary prediction model used for comparaisons.

    model_labels : dict, optional
        Name of the `main`, `compare`, `generator` model.

    generator : sklearn.estimator, optional
        Underlying generator. If not `None` will plot its own predictions.

    is_plot_real : bool, optional
        Whether to plot the underlying `Y_trgt`.

    train_min_max : tuple of float, optional
        Min and maximum boundary used during training. Important to unscale X to
        its actual values (i.e. plot will not be in -1,1).

    is_fill_generator_std : bool, optional
        Whether to show the generator's std filled in or simply the outline.

    y_lim : tuple of int, optional
        Min max y limit. If one is None, then auto.

    plot_config_kwargs : dict, optional
        Other arguments to `plot_config_kwargs`.

    kwargs :
        Additional arguments to `_plot_posterior_predefined_cntxt`.
    """
    with plot_config(plot_config_kwargs):
        set_seed(seed)

        _check_input(X, Y)
        X_cntxt, Y_cntxt, X_trgt, Y_trgt = get_cntxt_trgt(X, Y)

        alpha_init = 1 if compare_model is None else 0.5

        ax = _plot_posterior_predefined_cntxt(
            model,
            X_cntxt,
            Y_cntxt,
            X_trgt,
            train_min_max=train_min_max,
            Y_trgt=Y_trgt if is_plot_real else None,
            model_label=model_labels["main"],
            alpha_init=alpha_init,
            mean_std_colors=("b", "tab:blue"),
            ax=ax,
            **kwargs,
        )

        if compare_model is not None:
            ax = _plot_posterior_predefined_cntxt(
                compare_model,
                X_cntxt,
                Y_cntxt,
                X_trgt,
                train_min_max=train_min_max,
                model_label=model_labels["compare"],
                ax=ax,
                alpha_init=alpha_init,
                mean_std_colors=("m", "tab:pink"),
                **kwargs,
            )

        if generator is not None:
            X_cntxt_plot = rescale_range(X_cntxt, (-1, 1), train_min_max).numpy()[0]
            # clones so doesn't change real generator => can still sample prior
            generator = sklearn.base.clone(generator)
            generator.fit(X_cntxt_plot, Y_cntxt.numpy()[0])
            X_trgt_plot = rescale_range(X, (-1, 1), train_min_max).numpy()[0].flatten()
            mean_y, std_y = generator.predict(
                X_trgt_plot[:, np.newaxis], return_std=True
            )
            mean_y = mean_y.flatten()
            ax.plot(
                X_trgt_plot,
                mean_y,
                alpha=alpha_init / 1.5,
                c="g",
                label=model_labels["generator"],
            )

            if is_fill_generator_std:
                ax.fill_between(
                    X_trgt_plot,
                    mean_y - std_y,
                    mean_y + std_y,
                    alpha=alpha_init / 10,
                    color="tab:green",
                )
            else:
                ax.plot(
                    X_trgt_plot,
                    mean_y - std_y,
                    alpha=alpha_init / 2,
                    c="g",
                    linestyle="--",
                )
                ax.plot(
                    X_trgt_plot,
                    mean_y + std_y,
                    alpha=alpha_init / 2,
                    c="g",
                    linestyle="--",
                )

            ax.legend()
            ax.set_ylim([y_lim[0], y_lim[1]])

    return ax


def _check_input(*inp):
    for x in inp:
        if (x is not None) and not (x.dim() == 3 and x.dim() == 3 and x.shape[0] == 1):
            raise ValueError(
                "input should have 3 dim with first (batch size) of 0, but `x.shape={}`.".format(
                    x.shape
                )
            )


def _rescale_ylim(y_min, y_max):
    """Make the y_lim range larger."""
    if y_min < 0:
        y_min *= 1.2
    else:
        y_min /= 0.9

    if y_max > 0:
        y_max *= 1.2
    else:
        y_max /= 0.9
    return y_min, y_max


def gen_p_y_pred(model, X_cntxt, Y_cntxt, X_trgt, n_samples):
    """Get the estimated (conditional) posterior predictive from a model."""
    if X_cntxt is None:
        X_cntxt = torch.zeros(1, 0, model.x_dim)
        Y_cntxt = torch.zeros(1, 0, model.y_dim)

    if isinstance(model, LatentNeuralProcessFamily):
        old_n_z_samples_test = model.n_z_samples_test
        model.n_z_samples_test = n_samples
        p_yCc, *_ = model.forward(X_cntxt, Y_cntxt, X_trgt)
        model.n_z_samples_test = old_n_z_samples_test

    else:
        # can only use one sample in this case
        p_yCc, *_ = model.forward(X_cntxt, Y_cntxt, X_trgt)

    mean_ys = p_yCc.base_dist.loc.detach().numpy()
    std_ys = p_yCc.base_dist.scale.detach().numpy()

    for i in range(mean_ys.shape[0]):
        yield mean_ys[i, 0, :, 0].flatten(), std_ys[i, 0, :, 0].flatten()

    """
    else:
        z_sample = torch.randn((1, model.r_dim))
        r = z_sample.unsqueeze(1).expand(1, X_trgt.size(1), model.r_dim)
        dec_input = model.make_dec_inp(r, z_sample, X_trgt)
        p_y_pred = model.decode(dec_input, X_trgt)
    
    # don't return n_z_samples so that same 
    return p_yCc[0]
    """


def _plot_posterior_predefined_cntxt(
    model,
    X_cntxt,
    Y_cntxt,
    X_trgt,
    Y_trgt=None,
    n_samples=1,
    is_plot_std=False,
    train_min_max=(-2, 2),
    model_label="Model",
    alpha_init=1,
    mean_std_colors=("b", "tab:blue"),
    title=None,
    figsize=DFLT_FIGSIZE,
    ax=None,
):
    """
    Plot (samples from) the conditional posterior predictive estimated by a model.

    Parameters
    ----------
    model : nn.Module

    X_cntxt: torch.Tensor, size=[1, n_cntxt, x_dim]
        Set of all context features {x_i}.

    Y_cntxt: torch.Tensor, size=[1, n_cntxt, y_dim]
        Set of all context values {y_i}.

    X_trgt: torch.Tensor, size=[1, n_trgt, x_dim]
        Set of all target features {x_t}.

    Y_trgt: torch.Tensor, size=[1, n_trgt, y_dim], optional
        Set of all target values {y_t}. If not `None` plots the underlying function.

    n_samples : int, optional
        Number of samples from the posterior.

    is_plot_std : bool, optional
        Wheter to plot the predicted standard deviation.

    train_min_max : tuple of float, optional
        Min and maximum boundary used during training. Important to unscale X to
        its actual values (i.e. plot will not be in -1,1).

    alpha_init : float, optional
        Transparency level to use.

    mean_std_colors : tuple of str, optional
        Color of the predicted mean and std for plotting.

    model_label : str, optional
        Name of the model for the legend.

    title : str, optional

    figsize : tuple, optional

    ax : plt.axes.Axes, optional
    """

    _check_input(X_cntxt, Y_cntxt, X_trgt)

    mean_color, std_color = mean_std_colors

    is_conditioned = X_cntxt is not None  # plot posterior instead prior

    model.eval()
    model = model.cpu()

    X_trgt_plot = X_trgt.numpy()[0].flatten()
    X_interp = (X_trgt_plot > -1) & (X_trgt_plot < 1)
    # input to model should always be between -1 1 but not for plotting
    X_trgt_plot = rescale_range(X_trgt_plot, (-1, 1), train_min_max)

    x_min = min(X_trgt_plot)
    x_max = max(X_trgt_plot)

    if is_conditioned:
        X_cntxt_plot = X_cntxt.numpy()[0].flatten()
        X_cntxt_plot = rescale_range(X_cntxt_plot, (-1, 1), train_min_max)

    # make alpha dependent on number of samples
    alpha = alpha_init / (n_samples) ** 0.5

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        y_min, y_max = 0, 0
    else:
        y_min, y_max = ax.get_ylim()

    for i, (mean_y, std_y) in enumerate(
        gen_p_y_pred(model, X_cntxt, Y_cntxt, X_trgt, n_samples)
    ):

        if i == 0:
            # only add single label
            ax.plot(
                X_trgt_plot, mean_y, alpha=alpha, c=mean_color, label=f"{model_label}",
            )
        else:
            ax.plot(X_trgt_plot, mean_y, alpha=alpha, c=mean_color)

        if is_plot_std:
            ax.fill_between(
                X_trgt_plot,
                mean_y - std_y,
                mean_y + std_y,
                alpha=alpha / 7,
                color=std_color,
            )
            y_min = min(y_min, (mean_y - std_y)[X_interp].min())
            y_max = max(y_max, (mean_y + std_y)[X_interp].max())
        else:
            y_min = min(y_min, (mean_y)[X_interp].min())
            y_max = max(y_max, (mean_y)[X_interp].max())

    if Y_trgt is not None:
        _check_input(Y_trgt)
        X_trgt = X_trgt.numpy()[0].flatten()
        Y_trgt = Y_trgt.numpy()[0, :, 0].flatten()
        X_trgt_plot = rescale_range(X_trgt, (-1, 1), train_min_max)
        ax.plot(X_trgt_plot, Y_trgt, "--k", alpha=0.7, label="Target Function")
        y_min = min(y_min, Y_trgt.min())
        y_max = max(y_max, Y_trgt.max())

    if is_conditioned:
        ax.scatter(X_cntxt_plot, Y_cntxt[0, :, 0].numpy(), c="k")
        x_min = min(min(X_cntxt_plot), x_min)
        x_max = max(max(X_cntxt_plot), x_max)

    ax.set_xlim(x_min, x_max)

    # extrapolation might give huge values => rescale to have y lim as interpolation
    ax.set_ylim(_rescale_ylim(y_min, y_max))

    if x_max > train_min_max[1]:  # right extrapolation
        ax.axvline(
            x=train_min_max[1],
            color="r",
            linestyle=":",
            alpha=alpha_init / 2,
            label="Extrapolation Boundary",
        )

    if x_min < train_min_max[0]:  # left extrapolation
        ax.axvline(
            x=train_min_max[0],
            color="r",
            linestyle=":",
            alpha=alpha_init / 2,
            label="Extrapolation Boundary",
        )

    if title is not None:
        ax.set_title(title, fontsize=14)

    ax.legend()

    return ax
