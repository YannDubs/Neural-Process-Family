import copy
import os
import sys
from functools import partial

import matplotlib.pyplot as plt
import skorch
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Matern,
    WhiteKernel,
)
from skorch import NeuralNet

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
from utils.data import DIR_DATA, GPDataset, get_train_test_img_dataset
from utils.data.helpers import DatasetMerger
from utils.data.imgs import SingleImage, get_test_upscale_factor
from utils.visualize import (
    plot_config,
    plot_posterior_samples,
    plot_posterior_samples_1d,
    plot_prior_samples_1d,
)


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
    kernels, save_file=f"{os.path.join(DIR_DATA, 'gp_dataset.hdf5')}", **kwargs,
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
                    resolution_factor=n_cntxt, upscale_factor=upscale_factor,
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
    plot_config_kwargs={},
    title="Model : {model_name} | Data : {data_name} | Num. Context : {n_cntxt}",
    left_extrap=0,
    right_extrap=0,
    pretty_renamer=PRETTY_RENAMER,
    is_plot_generator=True,
    **kwargs,
):
    """Plot posterior samples conditioned on `n_cntxt` context points for a set of trained trainers."""

    with plot_config(**plot_config_kwargs):
        n_trainers = len(trainers)

        fig, axes = plt.subplots(
            n_trainers, 1, figsize=(8, 3 * n_trainers), sharex=True, squeeze=False
        )

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
                n_samples=1, n_points=3 * dataset.n_points, test_min_max=test_min_max
            )  # use higher density for plotting

            plot_posterior_samples_1d(
                X,
                Y,
                get_n_cntxt(n_cntxt),
                trainer.module_,
                generator=dataset.generator if is_plot_generator else None,
                train_min_max=dataset.min_max,
                title=curr_title,
                ax=axes.flatten()[i],
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
