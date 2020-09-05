import copy
import os
import sys
import types
from functools import partial

import imageio
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Matern,
    WhiteKernel,
)

import skorch
import torch
from npf import GridConvCNP
from npf.architectures import MLP, merge_flat_input
from npf.utils.datasplit import (
    CntxtTrgtGetter,
    GetRandomIndcs,
    GridCntxtTrgtGetter,
    RandomMasker,
    SuperresolutionCntxtTrgtGetter,
    get_all_indcs,
    half_masker,
    no_masker,
)
from skorch import NeuralNet
from utils.data import DIR_DATA, GPDataset, get_train_test_img_dataset
from utils.data.helpers import DatasetMerger
from utils.data.imgs import SingleImage, get_test_upscale_factor
from utils.helpers import set_seed
from utils.visualize import (
    plot_config,
    plot_posterior_samples,
    plot_posterior_samples_1d,
    plot_prior_samples_1d,
)
from utils.visualize.helpers import fig2img
from utils.visualize.visualize_1d import _plot_posterior_predefined_cntxt

try:
    from pygifsicle import optimize
except ImportError:
    pass

# DATA
def get_img_datasets(datasets):
    """Return the correct instantiated train and test datasets."""
    train_datasets, test_datasets = dict(), dict()
    for d in datasets:
        train_datasets[d], test_datasets[d] = get_train_test_img_dataset(d)

    return train_datasets, test_datasets


def get_all_gp_datasets():
    """Return train / tets / valid sets for all GP experiments."""
    datasets, test_datasets, valid_datasets = dict(), dict(), dict()

    for f in [
        get_datasets_single_gp,
        get_datasets_variable_hyp_gp,
        get_datasets_variable_kernel_gp,
    ]:
        _datasets, _test_datasets, _valid_datasets = f()
        datasets.update(_datasets)
        test_datasets.update(_test_datasets)
        valid_datasets.update(_valid_datasets)

    return datasets, test_datasets, valid_datasets


def get_all_gp_datasets_old():
    """Return train / tets / valid sets for all GP experiments."""
    datasets, test_datasets, valid_datasets = dict(), dict(), dict()

    for f in [
        get_datasets_single_gp,
        get_datasets_varying_hyp_gp,
        get_datasets_variable_kernel_gp,
    ]:
        _datasets, _test_datasets, _valid_datasets = f()
        datasets.update(_datasets)
        test_datasets.update(_test_datasets)
        valid_datasets.update(_valid_datasets)

    return datasets, test_datasets, valid_datasets


def get_datasets_single_gp():
    """Return train / tets / valid sets for 'Samples from a single GP'."""
    kernels = dict()

    kernels["RBF_Kernel"] = RBF(length_scale=(0.2))

    kernels["Periodic_Kernel"] = ExpSineSquared(length_scale=0.5, periodicity=0.5)

    # kernels["Matern_Kernel"] = Matern(length_scale=0.2, nu=1.5)

    kernels["Noisy_Matern_Kernel"] = WhiteKernel(noise_level=0.1) + Matern(
        length_scale=0.2, nu=1.5
    )

    return get_gp_datasets(
        kernels,
        is_vary_kernel_hyp=False,  # use a single hyperparameter per kernel
        n_samples=50000,  # number of different context-target sets
        n_points=128,  # size of target U context set for each sample
        is_reuse_across_epochs=False,  # never see the same example twice
    )


def get_datasets_variable_hyp_gp():
    """Return train / tets / valid sets for 'Samples from GPs with varying Kernel hyperparameters'."""
    kernels = dict()

    kernels["Variable_Matern_Kernel"] = Matern(length_scale_bounds=(0.01, 0.3), nu=1.5)

    return get_gp_datasets(
        kernels,
        is_vary_kernel_hyp=True,  # use a different hyp for each samples
        n_samples=50000,  # number of different context-target sets
        n_points=128,  # size of target U context set for each sample
        is_reuse_across_epochs=False,  # never see the same example twice
    )


def get_datasets_variable_kernel_gp():
    """Return train / tets / valid sets for 'Samples from GPs with varying Kernels'."""

    datasets, test_datasets, valid_datasets = get_datasets_single_gp()
    return (
        dict(All_Kernels=DatasetMerger(datasets.values())),
        dict(All_Kernels=DatasetMerger(test_datasets.values())),
        dict(All_Kernels=DatasetMerger(valid_datasets.values())),
    )


def sample_gp_dataset_like(dataset, **kwargs):
    """Wrap the output of `get_samples` in a gp dataset."""
    new_dataset = copy.deepcopy(dataset)
    new_dataset.set_samples_(*dataset.get_samples(**kwargs))
    return new_dataset


def get_gp_datasets(
    kernels, save_file=f"{os.path.join(DIR_DATA, 'gp_dataset.hdf5')}", **kwargs
):
    """
    Return a train, test and validation set for all the given kernels (dict).
    """
    datasets = dict()

    def get_save_file(name, save_file=save_file):
        if save_file is not None:
            save_file = (save_file, name)
        return save_file

    for name, kernel in kernels.items():
        datasets[name] = GPDataset(
            kernel=kernel, save_file=get_save_file(name), **kwargs
        )

    datasets_test = {
        k: sample_gp_dataset_like(
            dataset, save_file=get_save_file(k), idx_chunk=-1, n_samples=10000
        )
        for k, dataset in datasets.items()
    }

    datasets_valid = {
        k: sample_gp_dataset_like(
            dataset,
            save_file=get_save_file(k),
            idx_chunk=-2,
            n_samples=dataset.n_samples // 10,
        )
        for k, dataset in datasets.items()
    }

    return datasets, datasets_test, datasets_valid


# HELPERS
class StrFormatter:
    """Defult dictionary that takes a key dependent function.

    Parameters
    ----------
    exact_match : dict, optional
        Dictionary of strings that will be replaced by exact match.

    substring_replace : dict, optional
        Dictionary of substring that will be replaced if no exact_match. Order matters.
        Everything is title case at this point. None gets mapped to "".

    to_upper : list, optional
        Words to upper case.

    """

    def __init__(self, exact_match={}, subtring_replace={}, to_upper=[]):
        self.exact_match = exact_match
        self.subtring_replace = subtring_replace
        self.to_upper = to_upper

    def __getitem__(self, key):
        if not isinstance(key, str):
            return key

        if key in self.exact_match:
            return self.exact_match[key]

        key = key.title()

        for match, replace in self.subtring_replace.items():
            if replace is None:
                replace = ""
            key = key.replace(match, replace)

        for w in self.to_upper:
            key = key.replace(w, w.upper())

        return key


PRETTY_RENAMER = StrFormatter(
    exact_match={
        # "celeba64": "CelebA 64x64",
        # "celeba32": "CelebA 32x32",
        # "zs-multi-mnist": "Zero Shot Multi-MNIST",
        # "zsmms": "Zero Shot Multi-MNIST",
        "celeba64": "CelebA64",
        "celeba32": "CelebA32",
        "zs-multi-mnist": "ZSMM",
        "zsmms": "ZSMM",
        "ConvNPFXL_NllLNPF": r"ConvNP $\mathcal{L}_{ML}$",  # for paper
        "ConvNPFL_NllLNPF": r"ConvNP $\mathcal{L}_{ML}$",  # for paper
        "ConvNPFXL_ElboLNPF": r"ConvNP $\mathcal{L}_{NP}$",  # for paper
        "SelfAttnNPF_NllLNPF": r"ANP $\mathcal{L}_{ML}$",  # for paper
        "SelfAttnNPF_ElboLNPF": r"ANP $\mathcal{L}_{NP}$",  # for paper
    },
    subtring_replace={
        "_": " ",
        "Elbofalse": "MLE",
        "Elbotrue": "ELBO",
        "Latlbtrue": "LB_Z",
        "Latlbfalse": "",
        "Siglbtrue": "LB_P",
        "Siglbfalse": "",
        "Vhalf": "Vert. Half",
        "hhalf": "Horiz. Half",
        "Attncnp": "AttnCNP",
        "Convcnp": "ConvCNP",
        "Attnlnp": "AttnLNP",
        "Convlnp": "ConvLNP",
        "Selfattn": "Attn",
        "npf Cnpf": "CNP",
        "npf Nlllnpf": "LNP",
        "npf Islnpf": "LNP IS",
        "npf Elbolnpf": "LNP ELBO",
        "npfxl Cnpf": "CNP XL",
        "npfxl Nlllnpf": "LNP XL",
        "npfxl Islnpf": "LNP IS XL",
        "npfxl Elbolnpf": "LNP ELBO XL",
    },
    to_upper=["Mnist", "Svhn", "Cnp", "Lnp", "Rbf"],
)


def add_y_dim(models, datasets):
    """Add y_dim to all of the models depending on the dataset. Return a dictionary of data dependent models."""
    return {
        data_name: {
            model_name: partial(model, y_dim=data_train.shape[0])
            for model_name, model in models.items()
        }
        for data_name, data_train in datasets.items()
    }


def get_n_cntxt(n_cntxt, is_1d=True, upscale_factor=1):
    """Return a context target splitter with a fixed number of context points."""
    if is_1d:
        return CntxtTrgtGetter(
            contexts_getter=GetRandomIndcs(a=n_cntxt, b=n_cntxt),
            targets_getter=get_all_indcs,
            is_add_cntxts_to_trgts=False,
        )
    else:
        return GridCntxtTrgtGetter(
            context_masker=RandomMasker(a=n_cntxt, b=n_cntxt),
            target_masker=no_masker,
            is_add_cntxts_to_trgts=False,
            upscale_factor=upscale_factor,
        )


# TO DO : docstrings
def plot_multi_posterior_samples_imgs(
    trainers,
    datasets,
    n_cntxt,
    plot_config_kwargs={},
    title="{model_name} | {data_name} | C={n_cntxt}",
    pretty_renamer=PRETTY_RENAMER,
    n_plots=4,
    figsize=(3, 3),
    is_superresolution=False,
    **kwargs,
):
    with plot_config(**plot_config_kwargs):
        n_trainers = len(trainers)

        fig, axes = plt.subplots(
            1,
            n_trainers,
            figsize=(figsize[0] * n_plots, figsize[1] * n_trainers),
            squeeze=False,
        )

        for i, (k, trainer) in enumerate(trainers.items()):
            data_name = k.split("/")[0]
            model_name = k.split("/")[1]
            dataset = datasets[data_name]

            if isinstance(n_cntxt, float) and n_cntxt < 1:
                if is_superresolution:
                    n_cntxt_title = f"{int(dataset.shape[1]*n_cntxt)}x{int(dataset.shape[2]*n_cntxt)}"
                else:
                    n_cntxt_title = f"{100*n_cntxt:.1f}%"
            elif isinstance(n_cntxt, str):
                n_cntxt_title = pretty_renamer[n_cntxt]
            else:
                n_cntxt_title = n_cntxt

            curr_title = title.format(
                model_name=pretty_renamer[model_name],
                n_cntxt=n_cntxt_title,
                data_name=pretty_renamer[data_name],
            )

            upscale_factor = get_test_upscale_factor(data_name)
            if n_cntxt in ["vhalf", "hhalf"]:
                cntxt_trgt_getter = GridCntxtTrgtGetter(
                    context_masker=partial(
                        half_masker, dim=0 if n_cntxt == "hhalf" else 1
                    ),
                    upscale_factor=upscale_factor,
                )
            elif is_superresolution:
                cntxt_trgt_getter = SuperresolutionCntxtTrgtGetter(
                    resolution_factor=n_cntxt, upscale_factor=upscale_factor
                )
            else:
                cntxt_trgt_getter = get_n_cntxt(
                    n_cntxt, is_1d=False, upscale_factor=upscale_factor
                )

            plot_posterior_samples(
                dataset,
                cntxt_trgt_getter,
                trainer.module_.cpu(),
                is_uniform_grid=isinstance(trainer.module_, GridConvCNP),
                ax=axes.flatten()[i],
                n_plots=n_plots if not isinstance(dataset, SingleImage) else 1,
                is_mask_cntxt=not is_superresolution,
                **kwargs,
            )
            axes.flatten()[i].set_title(curr_title)

    return fig


# TO DO : docstrings
def plot_multi_posterior_samples_1d(
    trainers,
    datasets,
    n_cntxt,
    trainers_compare=None,
    plot_config_kwargs={},
    title="Model : {model_name} | Data : {data_name} | Num. Context : {n_cntxt}",
    left_extrap=0,
    right_extrap=0,
    pretty_renamer=PRETTY_RENAMER,
    is_plot_generator=True,
    imgsize=(8, 3),
    **kwargs,
):
    """Plot posterior samples conditioned on `n_cntxt` context points for a set of trained trainers."""

    with plot_config(**plot_config_kwargs):
        n_trainers = len(trainers)

        n_col = 1 if trainers_compare is None else 2
        fig, axes = plt.subplots(
            n_trainers,
            n_col,
            figsize=(imgsize[0] * n_col, imgsize[1] * n_trainers),
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        for j, curr_trainers in enumerate([trainers, trainers_compare]):
            if curr_trainers is None:
                continue

            for i, (k, trainer) in enumerate(trainers.items()):
                data_name = k.split("/")[0]
                model_name = k.split("/")[1]
                dataset = datasets[data_name]
                curr_title = title.format(
                    model_name=pretty_renamer[model_name],
                    n_cntxt=n_cntxt,
                    data_name=pretty_renamer[data_name],
                )

                test_min_max = dataset.min_max
                if (left_extrap != 0) or (right_extrap != 0):
                    test_min_max = (
                        dataset.min_max[0] - left_extrap,
                        dataset.min_max[1] + right_extrap,
                    )
                    trainer.module_.set_extrapolation(test_min_max)

                X, Y = dataset.get_samples(
                    n_samples=1,
                    n_points=3 * dataset.n_points,
                    test_min_max=test_min_max,
                )  # use higher density for plotting

                plot_posterior_samples_1d(
                    X,
                    Y,
                    get_n_cntxt(n_cntxt),
                    trainer.module_,
                    generator=dataset.generator if is_plot_generator else None,
                    train_min_max=dataset.min_max,
                    title=curr_title,
                    ax=axes[i, j],
                    **kwargs,
                )

        plt.tight_layout()

    return fig


def plot_multi_prior_samples_1d(trainers, datasets, **kwargs):
    """Plot prior samples conditioned on `n_cntxt` context points for a set of trained trainers."""
    n_trainers = len(trainers)

    fig, axes = plt.subplots(
        n_trainers, 1, figsize=(8, 3 * n_trainers), sharex=True, squeeze=False
    )

    for i, (k, trainer) in enumerate(trainers.items()):
        data_name = k.split("/")[0]
        model_name = k.split("/")[1].replace("_", " ")
        dataset = datasets[data_name]
        data_name = data_name.replace("_", " ")

        plot_prior_samples_1d(
            trainer.module_,
            title=f"{model_name} Trained Prior : {data_name}",
            train_min_max=dataset.min_max,
            ax=axes.flatten()[i],
            n_samples=1,
            is_plot_std=True,  # you cannot sample with CNP so use 1 sample but plot std
            **kwargs,
        )

    plt.tight_layout()

    return fig


def select_labels(dataset, label):
    """Return teh subset of the data with a specific label."""
    dataset = copy.deepcopy(dataset)
    filt = dataset.targets == label
    dataset.data = dataset.data[filt]
    dataset.targets = dataset.targets[filt]
    return dataset


# EXPLAINING GIF


def splitted_forward(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None):
    self._validate_inputs(X_cntxt, Y_cntxt, X_trgt, Y_trgt)

    # size = [batch_size, *n_cntxt, x_transf_dim]
    X_cntxt = self.x_encoder(X_cntxt)
    # size = [batch_size, *n_trgt, x_transf_dim]
    X_trgt = self.x_encoder(X_trgt)

    batch_size, n_cntxt, _ = X_cntxt.shape

    # size = [batch_size, n_induced, x_dim]
    X_induced = self._get_X_induced(X_cntxt)

    # don't resize because you want access to the density
    resizer = self.cntxt_to_induced.resizer
    self.cntxt_to_induced.resizer = torch.nn.Identity()

    # size = [batch_size, n_induced, value_size+1]
    R_induced = self.cntxt_to_induced(X_cntxt, X_induced, Y_cntxt)

    # reset resizer
    self.cntxt_to_induced.resizer = resizer

    # size = [batch_size, n_induced, r_dim]
    R = self.encode_globally(X_cntxt, Y_cntxt)

    # size = [n_z_samples, batch_size, *n_trgt, r_dim]
    R_trgt = self.trgt_dependent_representation(X_cntxt, None, R, X_trgt)

    # p(y|cntxt,trgt)
    # batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
    p_yCc = self.decode(X_trgt, R_trgt)

    return R_induced, R, R_trgt, p_yCc


def forward_Rinduced(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None):

    R_induced, *_ = self.splitted_forward(X_cntxt, Y_cntxt, X_trgt, Y_trgt=Y_trgt)

    n_induced = R_induced.size(1)
    to_burn = self.density_induced // 2

    # size = [1, batch_size, n_induced, y_dim]
    R_induced = R_induced[..., to_burn:-to_burn, :-1].unsqueeze(0)

    p_yCc = self.PredictiveDistribution(R_induced, torch.ones_like(R_induced))
    return p_yCc, None, None, None


def forward_density(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None):

    R_induced, *_ = self.splitted_forward(X_cntxt, Y_cntxt, X_trgt, Y_trgt=Y_trgt)

    n_induced = R_induced.size(1)
    to_burn = self.density_induced // 2

    # size = [1, batch_size, n_induced, 1]
    density = R_induced[..., to_burn:-to_burn, -1:].unsqueeze(0)

    p_yCc = self.PredictiveDistribution(density, torch.ones_like(density))
    return p_yCc, None, None, None


def get_forward_CNN(channel):
    def forward_CNN(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None):
        _, R_induced_post, *_ = self.splitted_forward(
            X_cntxt, Y_cntxt, X_trgt, Y_trgt=Y_trgt
        )

        n_induced = R_induced_post.size(1)
        to_burn = self.density_induced // 2

        # size = [1, batch_size, n_induced, y_dim]
        R_induced_post = R_induced_post[
            ..., to_burn:-to_burn, channel : channel + 1
        ].unsqueeze(0)

        p_yCc = self.PredictiveDistribution(
            R_induced_post, torch.ones_like(R_induced_post)
        )
        return p_yCc, None, None, None

    return forward_CNN


def get_forward_Rtrgt(channel):
    def forward_Rtrgt(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None):
        _, _, R_trgt, *_ = self.splitted_forward(
            X_cntxt, Y_cntxt, X_trgt, Y_trgt=Y_trgt
        )

        # size = [1, batch_size, n_induced, y_dim]
        R_trgt = R_trgt[..., channel : channel + 1]

        p_yCc = self.PredictiveDistribution(R_trgt, torch.ones_like(R_trgt))
        return p_yCc, None, None, None

    return forward_Rtrgt


def gif_explain(
    save_filename,
    dataset,
    model,
    plot_config_kwargs=dict(),
    seed=123,
    n_cntxt=10,
    fps=0.5,
    length_scale_delta=0 # increases length scale to make it clearer that smoothing out
):
    figs = []

    set_seed(seed)

    X, Y = dataset.get_samples(n_samples=1, n_points=1 * dataset.n_points)

    get_cntxt_trgt = get_n_cntxt(n_cntxt)
    X_cntxt, Y_cntxt, X_trgt, Y_trgt = get_cntxt_trgt(X, Y)
    model.splitted_forward = types.MethodType(splitted_forward, model)
    base_forward = model.forward  # store because will be modified at each step

    left, width = 0.25, 0.5
    bottom, height = 0.25, 0.5
    right = left + width
    top = bottom + height

    length_scale_param = model.cntxt_to_induced.radial_basis_func.length_scale_param
    model.cntxt_to_induced.radial_basis_func.length_scale_param = torch.nn.Parameter(length_scale_param + length_scale_delta)

    # Only points
    with plot_config(plot_config_kwargs):

        model.forward = types.MethodType(forward_Rinduced, model)

        _plot_posterior_predefined_cntxt(
            model,
            X_cntxt,
            Y_cntxt,
            torch.linspace(-1, 1, model.density_induced * 2).view(1, -1, 1),
            train_min_max=dataset.min_max,
            mean_std_colors=("tab:purple", "tab:purple"),
            is_plot_std=False,
            model_label="SetConv",
            linestyle="",
        )
        plt.gca().get_legend().remove()
        plt.xlim([-2, 2])

    figs.append(fig2img(plt.gcf()))
    plt.close()

    # Text : apply set conv
    with plot_config(plot_config_kwargs):

        model.forward = types.MethodType(forward_Rinduced, model)

        _plot_posterior_predefined_cntxt(
            model,
            X_cntxt,
            Y_cntxt,
            torch.linspace(-1, 1, model.density_induced * 2).view(1, -1, 1),
            train_min_max=dataset.min_max,
            mean_std_colors=("tab:purple", "tab:purple"),
            is_plot_std=False,
            model_label="SetConv",
            linestyle="",
            scatter_kwargs=dict(alpha=0.2),
        )
        plt.gca().get_legend().remove()
        plt.gca().text(
            0.5 * (left + right),
            0.5 * (bottom + top),
            "Apply SetConv",
            ha="center",
            va="center",
            fontsize=40,
            color="black",
            transform=plt.gca().transAxes,
            wrap=True,
        )
        plt.xlim([-2, 2])

    figs.append(fig2img(plt.gcf()))
    plt.close()

    #  set conv
    with plot_config(plot_config_kwargs):

        model.forward = types.MethodType(forward_Rinduced, model)

        _plot_posterior_predefined_cntxt(
            model,
            X_cntxt,
            Y_cntxt,
            torch.linspace(-1, 1, model.density_induced * 2).view(1, -1, 1),
            train_min_max=dataset.min_max,
            mean_std_colors=("tab:purple", "tab:purple"),
            is_plot_std=False,
            model_label="SetConv",
        )
        plt.xlim([-2, 2])

    figs.append(fig2img(plt.gcf()))
    plt.close()

    # Text : concatenate density
    with plot_config(plot_config_kwargs):

        model.forward = types.MethodType(forward_Rinduced, model)

        _plot_posterior_predefined_cntxt(
            model,
            X_cntxt,
            Y_cntxt,
            torch.linspace(-1, 1, model.density_induced * 2).view(1, -1, 1),
            train_min_max=dataset.min_max,
            mean_std_colors=("tab:purple", "tab:purple"),
            is_plot_std=False,
            model_label="SetConv",
            scatter_kwargs=dict(alpha=0.2),
            alpha_init=0.2,
        )

        plt.gca().text(
            0.5 * (left + right),
            0.5 * (bottom + top),
            "Concatenate Density",
            ha="center",
            va="center",
            fontsize=40,
            color="black",
            transform=plt.gca().transAxes,
            wrap=True,
        )

        plt.xlim([-2, 2])

    figs.append(fig2img(plt.gcf()))
    plt.close()

    # Density
    with plot_config(plot_config_kwargs):

        model.forward = types.MethodType(forward_Rinduced, model)

        _plot_posterior_predefined_cntxt(
            model,
            X_cntxt,
            Y_cntxt,
            torch.linspace(-1, 1, model.density_induced * 2).view(1, -1, 1),
            train_min_max=dataset.min_max,
            mean_std_colors=("tab:purple", "tab:purple"),
            is_plot_std=False,
            model_label="SetConv",
            scatter_kwargs=dict(alpha=0.0),
        )

        model.forward = types.MethodType(forward_density, model)

        _plot_posterior_predefined_cntxt(
            model,
            X_cntxt,
            Y_cntxt,
            torch.linspace(-1, 1, model.density_induced * 2).view(1, -1, 1),
            train_min_max=dataset.min_max,
            mean_std_colors=("tab:orange", "tab:orange"),
            is_plot_std=False,
            ax=plt.gca(),
            model_label="Density",
            scatter_kwargs=dict(alpha=0.0),
        )

        plt.xlim([-2, 2])

    figs.append(fig2img(plt.gcf()))
    plt.close()

    # Text : discretize
    with plot_config(plot_config_kwargs):

        model.forward = types.MethodType(forward_Rinduced, model)

        _plot_posterior_predefined_cntxt(
            model,
            X_cntxt,
            Y_cntxt,
            torch.linspace(-1, 1, model.density_induced * 2).view(1, -1, 1),
            train_min_max=dataset.min_max,
            mean_std_colors=("tab:purple", "tab:purple"),
            is_plot_std=False,
            model_label="SetConv",
            scatter_kwargs=dict(alpha=0.0),
            alpha_init=0.2,
        )

        model.forward = types.MethodType(forward_density, model)

        _plot_posterior_predefined_cntxt(
            model,
            X_cntxt,
            Y_cntxt,
            torch.linspace(-1, 1, model.density_induced * 2).view(1, -1, 1),
            train_min_max=dataset.min_max,
            mean_std_colors=("tab:orange", "tab:orange"),
            is_plot_std=False,
            ax=plt.gca(),
            model_label="Density",
            scatter_kwargs=dict(alpha=0.0),
            alpha_init=0.2,
        )

        plt.gca().text(
            0.5 * (left + right),
            0.5 * (bottom + top),
            "Discretize",
            ha="center",
            va="center",
            fontsize=40,
            color="black",
            transform=plt.gca().transAxes,
            wrap=True,
        )

        plt.xlim([-2, 2])

    figs.append(fig2img(plt.gcf()))
    plt.close()

    # discretize
    with plot_config(plot_config_kwargs):

        model.forward = types.MethodType(forward_Rinduced, model)

        _plot_posterior_predefined_cntxt(
            model,
            X_cntxt,
            Y_cntxt,
            torch.linspace(-1, 1, model.density_induced * 2).view(1, -1, 1),
            train_min_max=dataset.min_max,
            mean_std_colors=("tab:purple", "tab:purple"),
            is_plot_std=False,
            model_label="SetConv",
            is_smooth=False,
            marker=".",
            scatter_kwargs=dict(alpha=0.0),
        )

        model.forward = types.MethodType(forward_density, model)

        _plot_posterior_predefined_cntxt(
            model,
            X_cntxt,
            Y_cntxt,
            torch.linspace(-1, 1, model.density_induced * 2).view(1, -1, 1),
            train_min_max=dataset.min_max,
            mean_std_colors=("tab:orange", "tab:orange"),
            is_plot_std=False,
            ax=plt.gca(),
            model_label="Density",
            is_smooth=False,
            marker=".",
            scatter_kwargs=dict(alpha=0.0),
        )

        plt.xlim([-2, 2])

    figs.append(fig2img(plt.gcf()))
    plt.close()

    # text : apply CNN
    with plot_config(plot_config_kwargs):

        model.forward = types.MethodType(forward_Rinduced, model)

        _plot_posterior_predefined_cntxt(
            model,
            X_cntxt,
            Y_cntxt,
            torch.linspace(-1, 1, model.density_induced * 2).view(1, -1, 1),
            train_min_max=dataset.min_max,
            mean_std_colors=("tab:purple", "tab:purple"),
            is_plot_std=False,
            model_label="SetConv",
            is_smooth=False,
            marker=".",
            scatter_kwargs=dict(alpha=0.0),
            alpha_init=0.2,
        )

        model.forward = types.MethodType(forward_density, model)

        _plot_posterior_predefined_cntxt(
            model,
            X_cntxt,
            Y_cntxt,
            torch.linspace(-1, 1, model.density_induced * 2).view(1, -1, 1),
            train_min_max=dataset.min_max,
            mean_std_colors=("tab:orange", "tab:orange"),
            is_plot_std=False,
            ax=plt.gca(),
            model_label="Density",
            is_smooth=False,
            marker=".",
            scatter_kwargs=dict(alpha=0.0),
            alpha_init=0.2,
        )
        plt.gca().text(
            0.5 * (left + right),
            0.5 * (bottom + top),
            "Apply CNN",
            ha="center",
            va="center",
            fontsize=40,
            color="black",
            transform=plt.gca().transAxes,
            wrap=True,
        )
        plt.xlim([-2, 2])

    figs.append(fig2img(plt.gcf()))
    plt.close()
    model.cntxt_to_induced.radial_basis_func.length_scale_param = torch.nn.Parameter(length_scale_param)

    # apply CNN
    with plot_config(plot_config_kwargs):

        model.forward = types.MethodType(get_forward_CNN(1), model)

        _plot_posterior_predefined_cntxt(
            model,
            X_cntxt,
            Y_cntxt,
            torch.linspace(-1, 1, model.density_induced * 2).view(1, -1, 1),
            train_min_max=dataset.min_max,
            mean_std_colors=("tab:olive", ""),
            is_plot_std=False,
            model_label=f"Channel #{1}",
            is_smooth=False,
            marker=".",
            scatter_kwargs=dict(alpha=0.0),
        )

        for i, c in enumerate(["tab:cyan", "tab:pink"]):
            model.forward = types.MethodType(get_forward_CNN(i + 2), model)

            _plot_posterior_predefined_cntxt(
                model,
                X_cntxt,
                Y_cntxt,
                torch.linspace(-1, 1, model.density_induced * 2).view(1, -1, 1),
                train_min_max=dataset.min_max,
                mean_std_colors=(c, ""),  # secind color unused
                is_plot_std=False,
                model_label=f"Channel #{i+2}",
                is_smooth=False,
                marker=".",
                ax=plt.gca(),
                scatter_kwargs=dict(alpha=0.0),
            )

        plt.xlim([-2, 2])

    figs.append(fig2img(plt.gcf()))
    plt.close()

    # text : SetConv
    with plot_config(plot_config_kwargs):

        model.forward = types.MethodType(get_forward_CNN(1), model)

        _plot_posterior_predefined_cntxt(
            model,
            X_cntxt,
            Y_cntxt,
            torch.linspace(-1, 1, model.density_induced * 2).view(1, -1, 1),
            train_min_max=dataset.min_max,
            mean_std_colors=("tab:olive", ""),
            is_plot_std=False,
            model_label=f"Channel #{1}",
            is_smooth=False,
            marker=".",
            scatter_kwargs=dict(alpha=0.0),
            alpha_init=0.2,
        )

        for i, c in enumerate(["tab:cyan", "tab:pink"]):
            model.forward = types.MethodType(get_forward_CNN(i + 2), model)

            _plot_posterior_predefined_cntxt(
                model,
                X_cntxt,
                Y_cntxt,
                torch.linspace(-1, 1, model.density_induced * 2).view(1, -1, 1),
                train_min_max=dataset.min_max,
                mean_std_colors=(c, ""),  # secind color unused
                is_plot_std=False,
                model_label=f"Channel #{i+2}",
                is_smooth=False,
                marker=".",
                ax=plt.gca(),
                scatter_kwargs=dict(alpha=0.0),
                alpha_init=0.2,
            )

        plt.gca().text(
            0.5 * (left + right),
            0.5 * (bottom + top),
            "Apply SetConv",
            ha="center",
            va="center",
            fontsize=40,
            color="black",
            transform=plt.gca().transAxes,
            wrap=True,
        )

        plt.xlim([-2, 2])

    figs.append(fig2img(plt.gcf()))
    plt.close()

    # SetConv
    with plot_config(plot_config_kwargs):

        model.forward = types.MethodType(get_forward_Rtrgt(0), model)

        _plot_posterior_predefined_cntxt(
            model,
            X_cntxt,
            Y_cntxt,
            X_trgt,
            train_min_max=dataset.min_max,
            mean_std_colors=("tab:olive", ""),
            is_plot_std=False,
            model_label=f"Channel #{1}",
            scatter_kwargs=dict(alpha=0.0),
        )

        for i, c in enumerate(["tab:cyan", "tab:pink"]):
            model.forward = types.MethodType(get_forward_Rtrgt(i + 1), model)

            _plot_posterior_predefined_cntxt(
                model,
                X_cntxt,
                Y_cntxt,
                X_trgt,
                train_min_max=dataset.min_max,
                mean_std_colors=(c, "tab:blue"),  # secind color unused
                is_plot_std=False,
                model_label=f"Channel #{i+2}",
                ax=plt.gca(),
                scatter_kwargs=dict(alpha=0.0),
            )

        plt.xlim([-2, 2])

    figs.append(fig2img(plt.gcf()))
    plt.close()

    # text : Query target
    with plot_config(plot_config_kwargs):

        model.forward = types.MethodType(get_forward_Rtrgt(0), model)

        _plot_posterior_predefined_cntxt(
            model,
            X_cntxt,
            Y_cntxt,
            X_trgt,
            train_min_max=dataset.min_max,
            mean_std_colors=("tab:olive", ""),
            is_plot_std=False,
            model_label=f"Channel #{1}",
            scatter_kwargs=dict(alpha=0.0),
            alpha_init=0.2,
        )

        for i, c in enumerate(["tab:cyan", "tab:pink"]):
            model.forward = types.MethodType(get_forward_Rtrgt(i + 1), model)

            _plot_posterior_predefined_cntxt(
                model,
                X_cntxt,
                Y_cntxt,
                X_trgt,
                train_min_max=dataset.min_max,
                mean_std_colors=(c, "tab:blue"),  # secind color unused
                is_plot_std=False,
                model_label=f"Channel #{i+2}",
                ax=plt.gca(),
                scatter_kwargs=dict(alpha=0.0),
                alpha_init=0.2,
            )

        plt.gca().text(
            0.5 * (left + right),
            0.5 * (bottom + top),
            "Query Target Location",
            ha="center",
            va="center",
            fontsize=40,
            color="black",
            transform=plt.gca().transAxes,
            wrap=True,
        )

        plt.xlim([-2, 2])

    figs.append(fig2img(plt.gcf()))
    plt.close()

    # Query target
    set_seed(seed)
    X_trgt, Y_trgt, _, _ = get_n_cntxt(5)(X, Y)
    with plot_config(plot_config_kwargs):

        model.forward = types.MethodType(get_forward_Rtrgt(0), model)

        _plot_posterior_predefined_cntxt(
            model,
            X_cntxt,
            Y_cntxt,
            X_trgt,
            train_min_max=dataset.min_max,
            mean_std_colors=("tab:olive", ""),
            is_plot_std=False,
            model_label=f"Channel #{1}",
            scatter_kwargs=dict(alpha=0.0),
            is_smooth=False,
            marker="s",
        )

        for i, c in enumerate(["tab:cyan", "tab:pink"]):
            model.forward = types.MethodType(get_forward_Rtrgt(i + 1), model)

            _plot_posterior_predefined_cntxt(
                model,
                X_cntxt,
                Y_cntxt,
                X_trgt,
                train_min_max=dataset.min_max,
                mean_std_colors=(c, "tab:blue"),  # secind color unused
                is_plot_std=False,
                model_label=f"Channel #{i+2}",
                ax=plt.gca(),
                scatter_kwargs=dict(alpha=0.0),
                is_smooth=False,
                marker="s",
            )

        plt.xlim([-2, 2])

    figs.append(fig2img(plt.gcf()))
    plt.close()

    # text : gaussian
    set_seed(seed)
    X_trgt, Y_trgt, _, _ = get_n_cntxt(5)(X, Y)
    with plot_config(plot_config_kwargs):

        model.forward = types.MethodType(get_forward_Rtrgt(0), model)

        _plot_posterior_predefined_cntxt(
            model,
            X_cntxt,
            Y_cntxt,
            X_trgt,
            train_min_max=dataset.min_max,
            mean_std_colors=("tab:olive", ""),
            is_plot_std=False,
            model_label=f"Channel #{1}",
            scatter_kwargs=dict(alpha=0.0),
            is_smooth=False,
            marker="s",
            alpha_init=0.2,
        )

        for i, c in enumerate(["tab:cyan", "tab:pink"]):
            model.forward = types.MethodType(get_forward_Rtrgt(i + 1), model)

            _plot_posterior_predefined_cntxt(
                model,
                X_cntxt,
                Y_cntxt,
                X_trgt,
                train_min_max=dataset.min_max,
                mean_std_colors=(c, "tab:blue"),  # secind color unused
                is_plot_std=False,
                model_label=f"Channel #{i+2}",
                ax=plt.gca(),
                scatter_kwargs=dict(alpha=0.0),
                is_smooth=False,
                marker="s",
                alpha_init=0.2,
            )

        plt.gca().text(
            0.5 * (left + right),
            0.5 * (bottom + top),
            "Predict $\mu^{(t)},\sigma^{(t)}$ with a MLP",
            ha="center",
            va="center",
            fontsize=40,
            color="black",
            transform=plt.gca().transAxes,
            wrap=True,
        )

        plt.xlim([-2, 2])

    figs.append(fig2img(plt.gcf()))
    plt.close()

    # predict
    for n in [5, 20, 50, 100]:
        with plot_config(plot_config_kwargs):
            model.forward = base_forward

            set_seed(seed)
            X_trgt, Y_trgt, _, _ = get_n_cntxt(n)(X, Y)
            _plot_posterior_predefined_cntxt(
                model,
                X_cntxt,
                Y_cntxt,
                X_trgt,
                train_min_max=dataset.min_max,
                mean_std_colors=("b", "tab:blue"),
                is_plot_std=True,
                model_label=f"Prediction",
                scatter_kwargs=dict(alpha=0.0),
                is_smooth=False,
            )

            plt.gca().text(
                0.5 * (left + right),
                0.9 * (bottom + top),
                f"# targets = {n}",
                ha="center",
                va="center",
                fontsize=15,
                color="black",
                transform=plt.gca().transAxes,
                wrap=True,
            )

            plt.xlim([-2, 2])

        figs.append(fig2img(plt.gcf()))
        plt.close()

    # infinite predict
    with plot_config(plot_config_kwargs):
        model.forward = base_forward

        set_seed(seed)
        _, _, X_trgt, Y_trgt = get_n_cntxt(n)(X, Y)
        _plot_posterior_predefined_cntxt(
            model,
            X_cntxt,
            Y_cntxt,
            X_trgt,
            train_min_max=dataset.min_max,
            mean_std_colors=("b", "tab:blue"),
            is_plot_std=True,
            model_label=f"Prediction",
            scatter_kwargs=dict(alpha=0.0),
            is_smooth=True,
        )

        plt.gca().text(
            0.5 * (left + right),
            0.9 * (bottom + top),
            r"# targets = $\infty$",
            ha="center",
            va="center",
            fontsize=15,
            color="black",
            transform=plt.gca().transAxes,
            wrap=True,
        )

        plt.xlim([-2, 2])

    figs.append(fig2img(plt.gcf()))
    plt.close()

    imageio.mimsave(save_filename, figs, fps=fps)
    try:
        optimize(save_filename)
    except:
        pass
