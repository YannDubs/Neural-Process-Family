import logging
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
from scipy.interpolate import griddata
from skorch.dataset import unpack_data
from torchvision.utils import make_grid

from npf import GridConvCNP, LatentNeuralProcessFamily
from npf.utils.datasplit import GridCntxtTrgtGetter
from npf.utils.helpers import MultivariateNormalDiag, channels_to_2nd_dim, prod
from npf.utils.predict import SamplePredictor
from utils.data import cntxt_trgt_collate
from utils.helpers import mean, set_seed, tuple_cont_to_cont_tuple
from utils.train import EVAL_FILENAME

__all__ = [
    "plot_dataset_samples_imgs",
    "plot_qualitative_with_kde",
    "get_posterior_samples",
    "plot_posterior_samples",
    "plot_img_marginal_pred",
]

DFLT_FIGSIZE = (17, 9)
logger = logging.getLogger(__name__)


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


def get_posterior_samples(
    data,
    get_cntxt_trgt,
    model,
    is_uniform_grid=True,
    img_indcs=None,
    n_plots=4,
    seed=123,
    n_samples=3,  # if None selects all
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
            img_indcs = [random.randint(0, len(data) - 1) for _ in range(n_plots)]
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

    y_pred = SamplePredictor(model, is_dist=True)(mask_cntxt, Y_cntxt, mask_trgt)

    if is_select_different:
        # select the most different in average pixel L2 distance
        keep_most_different_samples_(y_pred, n_samples)
    elif isinstance(n_samples, int):
        # select first n_samples
        y_pred.base_dist.loc = y_pred.base_dist.loc[:n_samples, ...]
        y_pred.base_dist.scale = y_pred.base_dist.scale[:n_samples, ...]
    elif n_samples is None:
        pass  # select all
    else:
        ValueError(f"Unkown n_samples={n_samples}.")

    return y_pred, mask_cntxt, Y_cntxt, mask_trgt


# TO DOC
def plot_img_marginal_pred(
    model,
    data,
    get_cntxt_trgt,
    figsize=(11, 5),
    n_samples=5,
    is_uniform_grid=True,
    seed=123,
    n_plots_loop=1,
    wspace=0.3,
    n_marginals=5,
    n_columns=2,
    **kwargs,
):
    f, (ax0, ax1) = plt.subplots(
        1, 2, gridspec_kw={"width_ratios": [1, 1], "wspace": wspace}, figsize=figsize
    )

    predictive_all, mask_cntxt, X, mask_trgt = get_posterior_samples(
        data,
        get_cntxt_trgt,
        model,
        n_plots=n_plots_loop,
        is_uniform_grid=is_uniform_grid,
        seed=seed,
        n_samples=None,
    )

    if predictive_all.base_dist.loc.shape[0] == 1:
        logger.warning(
            "There's only a single sample of the posterior predictive, we treat it as the marginal."
        )

    arange = torch.linspace(0, 1, 1000)
    if is_uniform_grid:
        arange_marg = arange.view(1, 1000, 1, 1, 1)
    else:
        arange_marg = arange.view(1, 1000, 1, 1)

    best = float("inf")
    for i in range(n_plots_loop):
        predictive = MultivariateNormalDiag(
            predictive_all.base_dist.loc[:, i : i + 1, ...],
            predictive_all.base_dist.scale[:, i : i + 1, ...],
        )

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
            best_pred = predictive
            best_pred.base_dist.loc = predictive.base_dist.loc[:n_samples, ...]
            best_pred.base_dist.scale = predictive.base_dist.scale[:n_samples, ...]

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
        outs=[best_pred, best_mask_cntxt, best_X, best_mask_trgt],
        is_add_annot=False,
        n_plots=n_columns,
        **kwargs,
    )

    return f


def plot_posterior_samples(
    data,
    get_cntxt_trgt,
    model,
    is_uniform_grid=True,
    img_indcs=None,
    n_plots=4,
    imgsize=(7, 4),
    ax=None,
    seed=123,
    is_return=False,
    is_hrztl_cat=False,
    n_samples=1,
    outs=None,
    is_select_different=False,
    is_plot_std=False,
    interp_baselines=[],
    is_add_annot=True,
    rotate_annot=None,
    is_mask_cntxt=True,
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
        Model used for plotting.

    is_uniform_grid : bool, optional
        Whether the input are the image and corresponding masks rather than
        the slected pixels. Typically used for `GridConvCNP`.

    img_indcs : list of int, optional
        Indices of the images to plot. If `None` will randomly sample `n_plots`
        of them.

    n_plots : int, optional
        Number of images to samples. They will be plotted in different columns.
        Only used if `img_indcs` is `None`.

    imgsize : tuple, optional
        Figsize for each subimage. Will be multiplied by the number of plotted images.

    ax : plt.axes.Axes, optional

    seed : int, optional

    is_return : bool, optional
        Whether to return the grid instead of plotting it.

    is_hrztl_cat : bool, optional
        Whether to concatenate the plots horizontally instead of vertically. Only works well for
        n_plots=1.

    n_samples : int, optional
        Number of samples to plot.

    outs : tuple of tensors, optional
        Samples `(y_pred, mask_cntxt, Y_cntxt, mask_trgt)` to plot instead of using `get_posterior_samples`.

    is_select_different : bool, optional
        Whether to select the `n_samples` most different samples (in average L2 dist) instead of random.

    is_plot_std : bool, optional
        Whether to plot the standard deviation of the posterior predictive instead of only the mean.
        Note that the std is the average std across channels and is only shown for the last sample.

    interp_baselines : list of {"linear","nearest","cubic"}, optional
        List of interpolating baselines to plot in addition to the prediction from the model.

    is_add_annot : bool, optional
        Whether to add annotations *context, mean, ...).

    rotate_annot : float or {'vertical', 'horizontal'} or str, optional
        Rotation of annotation. If None automatic.

    is_mask_cntxt : bool, optional
        Whether to mask the context. If false plors the entire image, this is especially usefull
        when doing superresolution as all the image corresponds to the downscaled image. In this
        case, `get_cntxt_trgt` should be `SuperresolutionCntxtTrgtGetter`.
    """
    if outs is None:
        y_pred, mask_cntxt, X, mask_trgt = get_posterior_samples(
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
        y_pred, mask_cntxt, X, mask_trgt = outs

    if n_samples > 1 and not isinstance(model, LatentNeuralProcessFamily):
        if is_plot_std:
            raise ValueError("Cannot plot std when sampling from a CNPF.")
        # sampling for CNPFS
        mean_ys = y_pred.sample_n(n_samples)[:, 0, ...]
    else:
        mean_ys = y_pred.base_dist.loc

    mean_y = mean_ys[0]

    if n_samples > mean_ys.size(0):
        raise ValueError(
            f"Trying to plot more samples {n_samples} than the number of latent samples {mean_ys.size(0)}."
        )

    if isinstance(get_cntxt_trgt, dict):
        n_plots = get_cntxt_trgt["X_cntxt"].size(0)

    dim_grid = 2 if is_uniform_grid else 1
    if is_uniform_grid:
        mean_ys = mean_ys.view(n_samples, *X.shape)

    if X.shape[-1] == 1:
        X = X.expand(-1, *[-1] * dim_grid, 3)
        mean_ys = mean_ys.expand(n_samples, -1, *[-1] * dim_grid, 3)

    # make sure uses 3 channels
    std_ys = y_pred.base_dist.scale.expand(*mean_ys.shape)

    out_cntxt, mask_cntxt = get_img_toplot(
        data,
        X,
        mask_cntxt,
        is_uniform_grid,
        downscale_factor=get_downscale_factor(get_cntxt_trgt),
        is_mask=is_mask_cntxt,
    )

    outs = [out_cntxt]
    y_ticks_labels = ["Context"]

    for i in range(n_samples):
        out_pred, _ = get_img_toplot(
            data,
            mean_ys[i],
            mask_trgt,
            is_uniform_grid,
            downscale_factor=get_downscale_factor(get_cntxt_trgt),
        )
        outs.append(out_pred)

        if n_samples > 1:
            y_ticks_labels += [f"Sample {i+1}"]
        else:
            y_ticks_labels += [f"Pred. Mean"]

    if is_plot_std:
        out_std, _ = get_img_toplot(
            data,
            std_ys[n_samples - 1],  # only plot last std
            mask_trgt,
            is_uniform_grid,
            downscale_factor=get_downscale_factor(get_cntxt_trgt),
        )
        outs.append(out_std)

        if n_samples > 1:
            y_ticks_labels += [f"Std {n_samples}"]
        else:
            y_ticks_labels += [f"Pred. Std"]

    for interp in interp_baselines:
        # removing batch and channel from mask
        out_interps = []

        # loop over all context plots
        for i in range(mask_cntxt.shape[0]):
            single_mask_cntxt = mask_cntxt[i, :, :, 0]
            coord_y, coord_x = single_mask_cntxt.nonzero().unbind(1)
            grid_x, grid_y = np.meshgrid(
                np.arange(0, out_cntxt.shape[2]), np.arange(0, out_cntxt.shape[1])
            )
            out_interp = griddata(
                (coord_x, coord_y),
                out_cntxt[i, coord_y, coord_x],
                (grid_x, grid_y),
                method=interp,
            )
            out_interps.append(torch.from_numpy(out_interp))
        out_interp = torch.stack(out_interps, dim=0)
        outs.append(out_interp)

        y_ticks_labels += [f"{interp} Interp.".title()]

    outs = channels_to_2nd_dim(torch.cat(outs, dim=0)).detach()
    if is_hrztl_cat:
        tmp = []
        for i in range(n_plots):
            tmp.extend(outs[i::n_plots])
        outs = tmp

    n_fig_per_row = n_plots
    n_fig_per_col = len(y_ticks_labels)

    if is_hrztl_cat:
        n_fig_per_row, n_fig_per_col = n_fig_per_col, n_fig_per_row

    grid = make_grid(outs, nrow=n_fig_per_row, pad_value=1.0)

    if is_return:
        return grid

    if ax is None:
        figsize = (imgsize[0] * n_fig_per_row, imgsize[1] * n_fig_per_col)
        fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(grid.permute(1, 2, 0).numpy())

    if is_add_annot:
        idx_text = 2 if is_hrztl_cat else 1
        middle_img = data.shape[idx_text] // 2 + 1  # half height
        y_ticks = [middle_img * (2 * i + 1) for i in range(len(y_ticks_labels))]

        if is_hrztl_cat:
            if rotate_annot is None:
                rotate_annot = 20

            # to test
            ax.xaxis.set_major_locator(ticker.FixedLocator(y_ticks))
            ax.set_xticklabels(y_ticks_labels, rotation=rotate_annot, ha="right")
            ax.set_yticks([])

        else:
            if rotate_annot is None:
                rotate_annot = "vertical"

            ax.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))
            ax.set_yticklabels(y_ticks_labels, rotation=rotate_annot, va="center")
            ax.set_xticks([])

        remove_axis(ax)
    else:
        ax.axis("off")


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
    font_size=12,
    h_pad=-3,
    x_lim={},
    is_smallest_xrange=False,
    kdeplot_kwargs={},
    n_samples=1,
    upscale_factor=1,
    **kwargs,
):
    """
    Plot qualitative samples using `plot_posterior_samples` but select the samples and mask to plot
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

    is_smallest_xrange : bool, optional
        Whether to rescale the x axis based on the range of percentils.

    kdeplot_kwargs : dict, optional
        Additional arguments to `sns.kdeplot`

    upscale_factor : float, optional
        Whether to upscale the image => extrapolation. Only if not uniform grid.

    kwargs

    !VERY DIRTY
    """

    kwargs["n_samples"] = n_samples
    kwargs["is_plot_std"] = False
    kwargs["is_add_annot"] = False

    if percentiles is not None:
        n_images = len(percentiles)

    plt.rcParams.update({"font.size": font_size})
    fig, axes = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": height_ratios}
    )

    # a dictionary that has "upscale_factor" which is needed for downscaling when plotting
    # only is not grided
    CntxtTrgtDictUpscale = partial(CntxtTrgtDict, upscale_factor=upscale_factor)

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
        cntxt_trgt_getter = GridCntxtTrgtGetter(upscale_factor=upscale_factor)

        for i in range(n_images):
            X = selected_data["Y_cntxt"][i]
            X_cntxt, Y_cntxt = cntxt_trgt_getter.select(
                X, None, selected_data["X_cntxt"][i]
            )
            X_trgt, Y_trgt = cntxt_trgt_getter.select(
                X, None, selected_data["X_trgt"][i]
            )

            yield CntxtTrgtDictUpscale(
                X_cntxt=X_cntxt, Y_cntxt=Y_cntxt, X_trgt=X_trgt, Y_trgt=Y_trgt
            )

    def _plot_posterior_img_selected(name, trainer, selected_data, is_grided_trainer):
        is_uniform_grid = isinstance(trainer.module_, GridConvCNP)

        kwargs["img_indcs"] = []
        kwargs["is_uniform_grid"] = is_uniform_grid
        kwargs["is_return"] = True

        if not is_uniform_grid:

            if is_grided_trainer:
                grids = [
                    plot_posterior_samples(
                        dataset, data, trainer.module_.cpu(), **kwargs
                    )
                    for i, data in enumerate(_grid_to_points(selected_data))
                ]
            else:

                grids = [
                    plot_posterior_samples(
                        dataset,
                        CntxtTrgtDictUpscale(
                            **{k: v[i] for k, v in selected_data.items()}
                        ),
                        trainer.module_.cpu(),
                        **kwargs,
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
                        plot_posterior_samples(
                            dataset,
                            dict(
                                X_cntxt=X_cntxt,
                                Y_cntxt=Y_trgt,  # Y_trgt is all X because no masking for target (assumption)
                                X_trgt=X_trgt,
                                Y_trgt=Y_trgt,
                            ),
                            trainer.module_.cpu(),
                            **kwargs,
                        )
                    )

                grids = [g[..., 2:] if i != 0 else g for i, g in enumerate(grids)]

                return torch.cat(grids, axis=-1)
            else:
                return plot_posterior_samples(
                    dataset,
                    {k: torch.cat(v, dim=0) for k, v in selected_data.items()},
                    trainer.module_.cpu(),
                    **kwargs,
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


# HELPERS
def get_img_toplot(
    data, to_plot, mask, is_uniform_grid, downscale_factor=1, is_mask=True
):
    mask_toapply = mask if is_mask else torch.ones_like(mask).bool()
    dim_grid = 2 if is_uniform_grid else 1

    if is_uniform_grid:
        background = (
            data.missing_px_color.view(1, *[1] * dim_grid, 3)
            .expand(*to_plot.shape)
            .clone()
        )
        if mask.size(-1) == 1:
            out = torch.where(mask_toapply, to_plot, background)
        else:
            background[mask_toapply.squeeze(-1)] = to_plot.reshape(-1, 3)
            out = background.clone()

    else:
        out, _ = points_to_grid(
            mask_toapply,
            to_plot,
            data.shape[1:],
            background=data.missing_px_color,
            downscale_factor=downscale_factor,
        )

        # the second time is to get the mask (not the first one because in case not `is_mask`)
        # you still want to return the actual mask
        _, mask = points_to_grid(
            mask, to_plot, data.shape[1:], downscale_factor=downscale_factor
        )

    return out, mask


def keep_most_different_samples_(samples, n_samples, p=2):
    """Keep the `n_samples` most different samples (of the posterior predictive) using Lp distance of mean."""
    n_possible_samples = samples.batch_shape[0]
    assert n_samples <= n_possible_samples

    loc = samples.base_dist.loc
    scale = samples.base_dist.scale

    # select the first image to start with
    selected_idcs = [0]
    pool_idcs = set(range(1, n_possible_samples))

    for i in range(n_samples - 1):
        mean_distances = {
            i: mean(
                [
                    torch.dist(loc[selected_idx], loc[i], p=p)
                    for selected_idx in selected_idcs
                ]
            )
            for i in pool_idcs
        }
        idx_to_select = max(mean_distances, key=mean_distances.get)
        selected_idcs.append(idx_to_select)
        pool_idcs.remove(idx_to_select)

    samples.base_dist.loc = loc[selected_idcs]
    samples.base_dist.scale = scale[selected_idcs]


def marginal_log_like(predictive, samples):
    """compute log likelihood for evaluation"""
    # size = [n_z_samples, batch_size, *]
    log_p = predictive.log_prob(samples)

    # mean overlay samples in log space
    ll = torch.logsumexp(log_p, 0) - math.log(predictive.batch_shape[0])

    return ll.exp()


def sarle(out, axis=0):
    """Return sarle multi modal coefficient"""
    k = scipy.stats.kurtosis(out, axis=axis, fisher=True)
    g = scipy.stats.skew(out, axis=axis)
    n = out.shape[1]
    denom = k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 2))
    return (g ** 2 + 1) / denom


def get_downscale_factor(get_cntxt_trgt):
    """Return the scaling factor for the test set (used when extrapolation.)"""
    downscale_factor = 1
    try:
        downscale_factor = get_cntxt_trgt.upscale_factor
    except AttributeError:
        pass
    return downscale_factor


def remove_axis(ax, is_rm_ticks=True, is_rm_spines=True):
    """Remove all axis but not the labels."""
    if is_rm_spines:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_frame_on(False)

    if is_rm_ticks:
        ax.tick_params(bottom="off", left="off")


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


class CntxtTrgtDict(dict):
    """Dictionary that has `upscale_factor` argument."""

    def __init__(self, *arg, upscale_factor=1, **kw):
        self.upscale_factor = upscale_factor
        super().__init__(*arg, **kw)
