import math
import os
import random
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy
import seaborn as sns
import torch
from skorch.dataset import unpack_data
from torchvision.utils import make_grid

from npf import GridConvCNP
from npf.utils.datasplit import GridCntxtTrgtGetter
from npf.utils.helpers import MultivariateNormalDiag, channels_to_2nd_dim, prod
from npf.utils.predict import SamplePredictor, VanillaPredictor
from utils.data import cntxt_trgt_collate
from utils.helpers import set_seed, tuple_cont_to_cont_tuple
from utils.train import EVAL_FILENAME

__all__ = [
    "plot_dataset_samples_imgs",
    "plot_posterior_img",
    "plot_qualitative_with_kde",
    "get_posterior_samples",
    "plot_posterior_samples",
    "plot_img_marginal_pred",
]

DFLT_FIGSIZE = (17, 9)


def remove_axis(ax, is_rm_ticks=True, is_rm_spines=True):
    """Remove all axis but not the labels."""
    if is_rm_spines:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    if is_rm_ticks:
        ax.tick_params(bottom="off", left="off")


def plot_dataset_samples_imgs(
    dataset, n_plots=4, figsize=DFLT_FIGSIZE, ax=None, pad_value=1, seed=123, title=None
):
    """Plot `n_samples` samples of the a datset."""
    set_seed(seed)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    img_tensor = torch.stack(
        [dataset[random.randint(0, len(dataset) - 1)][0] for i in range(n_plots)], dim=0
    )
    grid = make_grid(img_tensor, nrow=2, pad_value=pad_value)

    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")

    if title is not None:
        ax.set_title(title)


def get_downscale_factor(get_cntxt_trgt):
    downscale_factor = 1
    try:
        downscale_factor = get_cntxt_trgt.test_upscale_factor
    except AttributeError:
        pass
    return downscale_factor


def plot_posterior_img(
    data,
    get_cntxt_trgt,
    model,
    MeanPredictor=VanillaPredictor,
    is_uniform_grid=True,
    img_indcs=None,
    n_plots=4,
    figsize=(18, 4),
    ax=None,
    seed=123,
    is_return=False,
    is_hrztl_cat=False,
):  # TO DOC
    """
    Plot the mean of the estimated posterior for images.

    Parameters
    ----------
    data : Dataset
        Dataset from which to sample the images.

    get_cntxt_trgt : callable or dict
        Function that takes as input the features and tagrets `X`, `y` and return
        the corresponding `X_cntxt, Y_cntxt, X_trgt, Y_trgt`. If dict should contain the correct 
        `X_cntxt, Y_cntxt, X_trgt, Y_trgt`.

    model : nn.Module
        Model used to initialize `MeanPredictor`.

    MeanPredictor : untitialized callable, optional
        Callable which is initalized with `MeanPredictor(model)` and then takes as
        input `X_cntxt, Y_cntxt, X_trgt` (`mask_cntxt, X, mask_trgt` if
        `is_uniform_grid`) and returns the mean the posterior. E.g. `VanillaPredictor`
        or `AutoregressivePredictor`.

    is_uniform_grid : bool, optional
        Whether the input are the image and corresponding masks rather than
        the slected pixels. Typically used for `GridConvCNP`.

    img_indcs : list of int, optional
        Indices of the images to plot. If `None` will randomly sample `n_plots`
        of them.

    n_plots : int, optional
        Number of images to samples. They will be plotted in different columns.
        Only used if `img_indcs` is `None`.

    figsize : tuple, optional

    ax : plt.axes.Axes, optional

    seed : int, optional
    """
    set_seed(seed)

    model.eval()

    dim_grid = 2 if is_uniform_grid else 1
    if isinstance(get_cntxt_trgt, dict):
        device = next(model.parameters()).device
        mask_cntxt = get_cntxt_trgt["X_cntxt"].to(device)
        X = get_cntxt_trgt["Y_cntxt"].to(device)
        mask_trgt = get_cntxt_trgt["X_trgt"].to(device)
        n_plots = mask_cntxt.size(0)

    else:
        if img_indcs is None:
            img_indcs = [random.randint(0, len(data)) for _ in range(n_plots)]
        n_plots = len(img_indcs)
        imgs = [data[i] for i in img_indcs]

        cntxt_trgt = cntxt_trgt_collate(
            get_cntxt_trgt, is_return_masks=is_uniform_grid
        )(imgs)[0]
        mask_cntxt, X, mask_trgt, _ = (
            cntxt_trgt["X_cntxt"],
            cntxt_trgt["Y_cntxt"],
            cntxt_trgt["X_trgt"],
            cntxt_trgt["Y_trgt"],
        )

    mean_y = MeanPredictor(model)(mask_cntxt, X, mask_trgt)

    if is_uniform_grid:
        mean_y = mean_y.view(*X.shape)

    if X.shape[-1] == 1:
        X = X.expand(-1, *[-1] * dim_grid, 3)
        mean_y = mean_y.expand(-1, *[-1] * dim_grid, 3)

    if is_uniform_grid:
        background = (
            data.missing_px_color.view(1, *[1] * dim_grid, 3)
            .expand(*mean_y.shape)
            .clone()
        )
        out_cntxt = torch.where(mask_cntxt, X, background)

        background[mask_trgt.squeeze(-1)] = mean_y.view(-1, 3)
        out_pred = background.clone()

    else:

        out_cntxt, _ = points_to_grid(
            mask_cntxt,
            X,
            data.shape[1:],
            background=data.missing_px_color,
            downscale_factor=get_downscale_factor(get_cntxt_trgt),
        )

        out_pred, _ = points_to_grid(
            mask_trgt,
            mean_y,
            data.shape[1:],
            background=data.missing_px_color,
            downscale_factor=get_downscale_factor(get_cntxt_trgt),
        )

    outs = [out_cntxt, out_pred]

    grid = make_grid(
        channels_to_2nd_dim(torch.cat(outs, dim=0)),
        nrow=n_plots * 2 if is_hrztl_cat else n_plots,
        pad_value=1.0,
    )

    if is_return:
        return grid

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")


def select_most_different_samples(samples, n_samples, p=2):
    """Select the `n_samples` most different sampels usingle Lp distance."""
    assert n_samples <= samples.size(0)

    selected_imgs = [samples[0]]
    pool_imgs = list(torch.unbind(samples[1:]))

    for i in range(n_samples - 1):
        mean_distances = [
            np.mean(
                [
                    torch.dist(img_in_selected, img_in_pool, p=p)
                    for img_in_selected in selected_imgs
                ]
            )
            for idx_in_pool, img_in_pool in enumerate(pool_imgs)
        ]
        idx_to_select = np.argmax(mean_distances)
        selected_imgs.append(pool_imgs.pop(idx_to_select))

    return torch.stack(selected_imgs)


def get_posterior_samples(
    data,
    get_cntxt_trgt,
    model,
    is_uniform_grid=True,
    img_indcs=None,
    n_plots=4,
    seed=123,
    n_samples=3,
    is_return_dist=False,
    is_select_different=False,
):

    set_seed(seed)

    model.eval()

    dim_grid = 2 if is_uniform_grid else 1
    if isinstance(get_cntxt_trgt, dict):
        device = next(model.parameters()).device
        mask_cntxt = get_cntxt_trgt["X_cntxt"].to(device)
        Y_cntxt = get_cntxt_trgt["Y_cntxt"].to(device)
        mask_trgt = get_cntxt_trgt["X_trgt"].to(device)
        # Y_trgt = get_cntxt_trgt["Y_trgt"].to(device)
        n_plots = mask_cntxt.size(0)

    else:
        if img_indcs is None:
            img_indcs = [random.randint(0, len(data)) for _ in range(n_plots)]
        n_plots = len(img_indcs)
        imgs = [data[i] for i in img_indcs]

        cntxt_trgt = cntxt_trgt_collate(
            get_cntxt_trgt, is_return_masks=is_uniform_grid
        )(imgs)[0]
        mask_cntxt, Y_cntxt, mask_trgt, _ = (
            cntxt_trgt["X_cntxt"],
            cntxt_trgt["Y_cntxt"],
            cntxt_trgt["X_trgt"],
            cntxt_trgt["Y_trgt"],
        )

    y_pred = SamplePredictor(model, is_dist=is_return_dist)(
        mask_cntxt, Y_cntxt, mask_trgt
    )

    # actually returns all samples when return dist
    if not is_return_dist:
        if is_select_different:
            # select the most different in average pixel L2 distance
            y_pred = select_most_different_samples(y_pred, n_samples)
        else:
            # select first n_samples
            y_pred = y_pred[:n_samples, ...]

    return y_pred, mask_cntxt, Y_cntxt, mask_trgt


def marginal_log_like(predictive, samples):
    # compute log likelihood for evaluation
    # size = [n_z_samples, batch_size, *]
    log_p = predictive.log_prob(samples)

    # mean overlay samples in log space
    ll = torch.logsumexp(log_p, 0) - math.log(predictive.batch_shape[0])

    return ll.exp()


def sarle(out, axis=0):
    k = scipy.stats.kurtosis(out, axis=axis, fisher=True)
    g = scipy.stats.skew(out, axis=axis)
    n = out.shape[1]
    denom = k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 2))
    return (g ** 2 + 1) / denom


def plot_img_marginal_pred(
    model,
    data,
    get_cntxt_trgt,
    figsize=(11, 5),
    n_samples=5,
    is_uniform_grid=True,
    seed=123,
    n_plots_loop=30,
    wspace=0.3,
    n_marginals=5,
    **kwargs
):
    f, (ax0, ax1) = plt.subplots(
        1, 2, gridspec_kw={"width_ratios": [1, 1], "wspace": wspace}, figsize=figsize,
    )

    predictive_all, mask_cntxt, X, mask_trgt = get_posterior_samples(
        data,
        get_cntxt_trgt,
        model,
        n_plots=n_plots_loop,
        is_uniform_grid=is_uniform_grid,
        seed=seed,
        is_return_dist=True,
    )

    best = float("inf")
    for i in range(n_plots_loop):
        predictive = MultivariateNormalDiag(
            predictive_all.base_dist.loc[:, i : i + 1, ...],
            predictive_all.base_dist.scale[:, i : i + 1, ...],
        )

        arange = torch.linspace(0, 1, 1000)
        if is_uniform_grid:
            arange_marg = arange.view(1, 1000, 1, 1, 1)
        else:
            arange_marg = arange.view(1, 1000, 1, 1)

        out = (
            marginal_log_like(predictive, arange_marg)
            .detach()
            .reshape(1000, -1)
            .numpy()
        )
        new_sarle = sarle(out)

        if np.median(new_sarle) < best:
            best_out = out
            best_sarles = new_sarle
            best_mean_ys = predictive.base_dist.loc.detach()[:n_samples, ...]
            best_mask_cntxt = mask_cntxt[i : i + 1, ...]
            best_X = X[i : i + 1, ...]
            best_mask_trgt = mask_trgt[i : i + 1, ...]
            best = np.median(best_sarles)

    idx = np.argsort(best_sarles)[:n_marginals]
    ax1.plot(arange, best_out[:, idx], alpha=0.7)

    sns.despine(top=True, right=True, left=False, bottom=False)
    ax1.set_yticks([])
    ax1.set_ylabel("Marginal Predictive")
    ax1.set_xlabel("Pixel Intensity")
    ax1.set_xlim(-0.1, 1)
    ax1.set_xticks([0, 0.5, 1])

    plot_posterior_samples(
        data,
        get_cntxt_trgt,
        model,
        is_uniform_grid=is_uniform_grid,
        seed=seed,
        n_samples=n_samples,
        ax=ax0,
        outs=[best_mean_ys, best_mask_cntxt, best_X, best_mask_trgt],
        **kwargs
    )


# should be merged with plot_posterior_imgs
def plot_posterior_samples(
    data,
    get_cntxt_trgt,
    model,
    is_uniform_grid=True,
    img_indcs=None,
    n_plots=4,
    figsize=(18, 4),
    ax=None,
    seed=123,
    is_return=False,  # DOC
    is_hrztl_cat=False,  # DOC
    n_samples=3,  # DOC
    outs=None,  # TO DOC
    is_select_different=False,
):
    """
    Plot the mean of the estimated posterior for images.

    Parameters
    ----------
    data : Dataset
        Dataset from which to sample the images.

    get_cntxt_trgt : callable or dict
        Function that takes as input the features and tagrets `X`, `y` and return
        the corresponding `X_cntxt, Y_cntxt, X_trgt, Y_trgt`. If dict should contain the correct 
        `X_cntxt, Y_cntxt, X_trgt, Y_trgt`.

    model : nn.Module
        Model used to initialize `MeanPredictor`.

    MeanPredictor : untitialized callable, optional
        Callable which is initalized with `MeanPredictor(model)` and then takes as
        input `X_cntxt, Y_cntxt, X_trgt` (`mask_cntxt, X, mask_trgt` if
        `is_uniform_grid`) and returns the mean the posterior. E.g. `VanillaPredictor`
        or `AutoregressivePredictor`.

    is_uniform_grid : bool, optional
        Whether the input are the image and corresponding masks rather than
        the slected pixels. Typically used for `GridConvCNP`.

    img_indcs : list of int, optional
        Indices of the images to plot. If `None` will randomly sample `n_plots`
        of them.

    n_plots : int, optional
        Number of images to samples. They will be plotted in different columns.
        Only used if `img_indcs` is `None`.

    figsize : tuple, optional

    ax : plt.axes.Axes, optional

    seed : int, optional

    is_select_different : bool, optional
        Whether to select the `n_samples` most different samples (in average L2 dist) instead of random.
    """
    if outs is None:
        mean_ys, mask_cntxt, X, mask_trgt = get_posterior_samples(
            data,
            get_cntxt_trgt,
            model,
            is_uniform_grid=is_uniform_grid,
            img_indcs=img_indcs,
            n_plots=n_plots,
            seed=seed,
            n_samples=n_samples,
            is_select_different=is_select_different,
        )
    else:
        mean_ys, mask_cntxt, X, mask_trgt = outs

    if isinstance(get_cntxt_trgt, dict):
        n_plots = get_cntxt_trgt["X_cntxt"].size(0)

    dim_grid = 2 if is_uniform_grid else 1
    if is_uniform_grid:
        mean_ys = mean_ys.view(n_samples, *X.shape)

    if X.shape[-1] == 1:
        X = X.expand(-1, *[-1] * dim_grid, 3)
        mean_ys = mean_ys.expand(n_samples, -1, *[-1] * dim_grid, 3)

    mean_y = mean_ys[0]

    if is_uniform_grid:
        background = (
            data.missing_px_color.view(1, *[1] * dim_grid, 3)
            .expand(*mean_y.shape)
            .clone()
        )
        out_cntxt = torch.where(mask_cntxt, X, background)

        background[mask_trgt.squeeze(-1)] = mean_y.view(-1, 3)
        out_pred = background.clone()

    else:
        out_cntxt, _ = points_to_grid(
            mask_cntxt,
            X,
            data.shape[1:],
            background=data.missing_px_color,
            downscale_factor=get_downscale_factor(get_cntxt_trgt),
        )

    outs = [out_cntxt]

    for i in range(n_samples):
        if is_uniform_grid:
            background = (
                data.missing_px_color.view(1, *[1] * dim_grid, 3)
                .expand(*mean_y.shape)
                .clone()
            )

            background[mask_trgt.squeeze(-1)] = mean_ys[i].view(-1, 3)
            out_pred = background.clone()

        else:

            out_pred, _ = points_to_grid(
                mask_trgt,
                mean_ys[i],
                data.shape[1:],
                background=data.missing_px_color,
                downscale_factor=get_downscale_factor(get_cntxt_trgt),
            )

        outs.append(out_pred)

    grid = make_grid(
        channels_to_2nd_dim(torch.cat(outs, dim=0)),
        nrow=n_plots * 2 if is_hrztl_cat else n_plots,
        pad_value=1.0,
    )

    if is_return:
        return grid

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")


class CntxtTrgtDict(dict):
    def __init__(self, *arg, test_upscale_factor=1, **kw):
        self.test_upscale_factor = test_upscale_factor
        super().__init__(*arg, **kw)


# TO CLEAN
def plot_qualitative_with_kde(
    named_trainer,
    dataset,
    named_trainer_compare=None,
    n_images=8,
    percentiles=None,  # if None uses uniform linspace from n_images
    figsize=DFLT_FIGSIZE,
    title=None,
    seed=123,
    height_ratios=[1, 3],
    font_size=14,
    h_pad=-3,
    x_lim={},
    is_smallest_xrange=False,
    kdeplot_kwargs={},
    n_samples=None,
    test_upscale_factor=1,
    **kwargs
):
    """
    Plot qualitative samples using `plot_posterior_img` but select the samples and mask to plot
    given the score at test time.

    Parameters
    ----------
    named_trainer : list [name, NeuralNet]
        Trainer (model outputted of training) and the name under which it should be displayed.

    dataset : 

    named_trainer_compare : list [name, NeuralNet], optional
        Like `named_trainer` but for a model against which to compare.

    n_images : int, optional
        Number of images to plot (at uniform interval of log like). Only used if `percentiles` is None.

    percentiles : list of float, optional
        Percentiles of log likelihood of the main model for which to select an image. The length
        of the list will correspond to the number fo images.

    figsize : tuple, optional

    title : str, optional

    seed : int, optional

    height_ratios : int iterable of length = nrows, optional
        Height ratios of the rows.
    
    font_size : int, optional
    
    h_pad : int, optional
        Padding between kde plot and images
    
    x_lim : dict, optional
        Dictionary containing one (or both) of "left", "right" correspomding to the x limit of kde plot.
    
    is_smallest_xrange=False,
    
    kdeplot_kwargs : dict, optional
        Additional arguments to `sns.kdeplot`

    test_upscale_factor : float, optional
        Whether to upscale the image => extrapolation. Only if not uniform grid.
    
    kwargs

    !VERY DIRTY
    """
    if n_samples is None:
        plot_posterior = plot_posterior_img
        n_samples = 1
    else:
        plot_posterior = partial(plot_posterior_samples, n_samples=n_samples)

    if percentiles is not None:
        n_images = len(percentiles)

    plt.rcParams.update({"font.size": font_size})
    fig, axes = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": height_ratios}
    )

    def _plot_kde_loglike(name, trainer):
        chckpnt_dirname = dict(trainer.callbacks_)["Checkpoint"].dirname
        test_eval_file = os.path.join(chckpnt_dirname, EVAL_FILENAME)
        test_loglike = np.loadtxt(test_eval_file, delimiter=",")
        sns.kdeplot(
            test_loglike, ax=axes[0], shade=True, label=name, cut=0, **kdeplot_kwargs
        )
        sns.despine()
        return test_loglike

    def _grid_to_points(selected_data):
        cntxt_trgt_getter = GridCntxtTrgtGetter(test_upscale_factor=test_upscale_factor)

        for i in range(n_images):
            X = selected_data["Y_cntxt"][i]
            X_cntxt, Y_cntxt = cntxt_trgt_getter.select(
                X, None, selected_data["X_cntxt"][i]
            )
            X_trgt, Y_trgt = cntxt_trgt_getter.select(
                X, None, selected_data["X_trgt"][i]
            )

            # a dictionary that has "test_upscale_factor" which is needed for downscaling when plotting
            yield CntxtTrgtDict(
                X_cntxt=X_cntxt,
                Y_cntxt=Y_cntxt,
                X_trgt=X_trgt,
                Y_trgt=Y_trgt,
                test_upscale_factor=test_upscale_factor,
            )

    def _plot_posterior_img_selected(name, trainer, selected_data, is_grided_trainer):
        is_uniform_grid = isinstance(trainer.module_, GridConvCNP)

        kwargs["img_indcs"] = []
        kwargs["is_uniform_grid"] = is_uniform_grid
        kwargs["is_return"] = True

        if not is_uniform_grid:

            if is_grided_trainer:
                grids = [
                    plot_posterior(dataset, data, trainer.module_.cpu(), **kwargs)
                    for i, data in enumerate(_grid_to_points(selected_data))
                ]
            else:
                grids = [
                    plot_posterior(
                        dataset,
                        {k: v[i] for k, v in selected_data.items()},
                        trainer.module_.cpu(),
                        **kwargs
                    )
                    for i in range(n_images)
                ]

            # images are padded by 2 pixels inbetween each but here you concatenate => will pad twice
            # => remove all the rleft padding for each besides first
            grids = [g[..., 2:] if i != 0 else g for i, g in enumerate(grids)]
            return torch.cat(grids, axis=-1)

        elif is_uniform_grid:
            if not is_grided_trainer:
                grids = []
                for i in range(n_images):

                    _, X_cntxt = points_to_grid(
                        selected_data["X_cntxt"][i],
                        selected_data["Y_cntxt"][i],
                        dataset.shape[1:],
                        background=torch.tensor([0.0] * dataset.shape[0]),
                    )
                    Y_trgt, X_trgt = points_to_grid(
                        selected_data["X_trgt"][i],
                        selected_data["Y_trgt"][i],
                        dataset.shape[1:],
                        background=torch.tensor([0.0] * dataset.shape[0]),
                    )

                    grids.append(
                        plot_posterior(
                            dataset,
                            dict(
                                X_cntxt=X_cntxt,
                                Y_cntxt=Y_trgt,  # Y_trgt is all X because no masking for target (assumption)
                                X_trgt=X_trgt,
                                Y_trgt=Y_trgt,
                            ),
                            trainer.module_.cpu(),
                            **kwargs
                        )
                    )

                grids = [g[..., 2:] if i != 0 else g for i, g in enumerate(grids)]

                return torch.cat(grids, axis=-1)
            else:
                return plot_posterior(
                    dataset,
                    {k: torch.cat(v, dim=0) for k, v in selected_data.items()},
                    trainer.module_.cpu(),
                    **kwargs
                )

    name, trainer = named_trainer
    test_loglike = _plot_kde_loglike(name, trainer)

    if named_trainer_compare is not None:
        left = axes[0].get_xlim()[0]
        _ = _plot_kde_loglike(*named_trainer_compare)
        axes[0].set_xlim(left=left)  # left bound by first model to not look strange

    if len(x_lim) != 0:
        axes[0].set_xlim(**x_lim)

    if percentiles is not None:
        idcs = []
        values = []
        for i, p in enumerate(percentiles):
            # value closest to percentile
            percentile_val = np.percentile(test_loglike, p, interpolation="nearest")
            idcs.append(np.argwhere(test_loglike == percentile_val).item())
            values.append(percentile_val)
        sorted_idcs = list(np.sort(idcs))[::-1]

        if is_smallest_xrange:
            axes[0].set_xlim(left=values[0] - 0.05, right=values[-1] + 0.05)
    else:
        # find indices such that same space between all
        values = np.linspace(test_loglike.min(), test_loglike.max(), n_images)
        idcs = [(np.abs(test_loglike - v)).argmin() for v in values]
        sorted_idcs = list(np.sort(idcs))[::-1]

    axes[0].set_ylabel("Density")
    axes[0].set_xlabel("Test Log-Likelihood")

    selected_data = []

    set_seed(seed)  # make sure same order and indices for cntxt and trgt
    i = -1

    saved_values = []
    queue = sorted_idcs.copy()
    next_idx = queue.pop()

    for data in trainer.get_iterator(dataset, training=False):
        Xi, yi = unpack_data(data)

        for cur_idx in range(yi.size(0)):
            i += 1
            if next_idx != i:
                continue

            selected_data.append(
                {k: v[cur_idx : cur_idx + 1, ...] for k, v in Xi.items()}
            )

            if len(queue) == 0:
                break
            else:
                next_idx = queue.pop()

    # puts back to non sorted array
    selected_data = [selected_data[sorted_idcs[::-1].index(idx)] for idx in idcs]

    selected_data = {k: v for k, v in tuple_cont_to_cont_tuple(selected_data).items()}

    for v in values:
        axes[0].axvline(v, linestyle=":", alpha=0.7, c="tab:green")

    axes[0].legend(loc="upper left")

    if title is not None:
        axes[0].set_title(title, fontsize=18)

    is_grided_trainer = isinstance(trainer.module_, GridConvCNP)
    grid = _plot_posterior_img_selected(name, trainer, selected_data, is_grided_trainer)

    middle_img = dataset.shape[1] // 2 + 1  # half height
    y_ticks = [middle_img, middle_img * 3]
    y_ticks_labels = ["Context", name]

    if named_trainer_compare is not None:
        grid_compare = _plot_posterior_img_selected(
            *named_trainer_compare, selected_data, is_grided_trainer
        )

        grid = torch.cat(
            (grid, grid_compare[:, grid_compare.size(1) // (n_samples + 1) + 1 :, :]),
            dim=1,
        )

        y_ticks += [middle_img * (3 + 2 * n_samples)]
        y_ticks_labels += [named_trainer_compare[0]]

    axes[1].imshow(grid.permute(1, 2, 0).numpy())

    axes[1].yaxis.set_major_locator(ticker.FixedLocator(y_ticks))
    axes[1].set_yticklabels(y_ticks_labels, rotation="vertical", va="center")

    remove_axis(axes[1])

    if percentiles is not None:
        axes[1].xaxis.set_major_locator(
            ticker.FixedLocator(
                [
                    (dataset.shape[2] // 2 + 1) * (i * 2 + 1)
                    for i, p in enumerate(percentiles)
                ]
            )
        )
        axes[1].set_xticklabels(["{}%".format(p) for p in percentiles])
    else:
        axes[1].set_xticks([])

    fig.tight_layout(h_pad=h_pad)
    # ----------------------------------------


def idcs_grid_to_idcs_flatten(idcs, grid_shape):
    """Convert a tensor containing indices of a grid to indices on the flatten grid."""
    for i, size in enumerate(grid_shape):
        idcs[:, :, i] *= prod(grid_shape[i + 1 :])
    return idcs.sum(-1)


def points_to_grid(
    X, Y, grid_shape, background=torch.tensor([0.0, 0.0, 0.0]), downscale_factor=1
):
    """Converts points to a grid (undo mask select from datasplit)"""

    batch_size, _, y_dim = Y.shape
    X = X.clone()
    background = background.view(1, *(1 for _ in grid_shape), y_dim).repeat(
        batch_size, *grid_shape, 1
    )

    X /= downscale_factor

    for i, size in enumerate(grid_shape):
        X[:, :, i] += 1  # in [0,2]
        X[:, :, i] /= 2 / (size - 1)  # in [0,size]

    X = X.round().long()
    idcs = idcs_grid_to_idcs_flatten(X, grid_shape)

    background = background.view(batch_size, -1, y_dim)
    mask = torch.zeros(batch_size, background.size(1), 1).bool()

    for b in range(batch_size):
        background[b, idcs[b], :] = Y[b]
        mask[b, idcs[b], :] = True

    background = background.view(batch_size, *grid_shape, y_dim)

    return background, mask.view(batch_size, *grid_shape, 1)
