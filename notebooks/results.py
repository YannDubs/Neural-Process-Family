""" isort:skip_file
"""
import copy
import contextlib
import logging
import os
from functools import partial
import sys
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skorch
import torch
from IPython.utils import io

sys.path.append("..")

from npf import NLLLossLNPF
from npf.utils.datasplit import (
    GridCntxtTrgtGetter,
    RandomMasker,
    half_masker,
    no_masker,
)
from train_imgs import get_model, main, parse_arguments
from utils.data import cntxt_trgt_collate, get_dataset, get_train_test_img_dataset
from utils.data.helpers import train_dev_split
from utils.helpers import count_parameters, load_all_results, set_seed, silent
from utils.train import _best_loss, train_models
from utils.visualize import (
    plot_dataset_samples_imgs,
    plot_losses,
    plot_qualitative_with_kde,
)
from utils.visualize.visualize_imgs import DFLT_FIGSIZE, points_to_grid


DIR = "results/neurips/"

get_cntxt_trgt = GridCntxtTrgtGetter(
    context_masker=RandomMasker(min_nnz=0.0, max_nnz=0.5),
    target_masker=no_masker,
    is_add_cntxts_to_trgts=False,
)


def select_labels(dataset, label):
    """Return teh subset of the data with a specific label."""
    dataset = copy.deepcopy(dataset)
    filt = dataset.targets == label
    dataset.data = dataset.data[filt]
    dataset.targets = dataset.targets[filt]
    return dataset


class StrFormatter:
    """Defult dictionary that takes a key dependent function.

    Parameters
    ----------
    exact_match : dict, optional
        Dictionary of strings that will be replaced by exact match.

    subtring_replace : dict, optional
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
        "celeba64": "CelebA 64x64",
        "celeba32": "CelebA 32x32",
        "zs-multi-mnist": "Zero Shot Multi-MNIST",
        "zsmms": "Zero Shot Multi-MNIST",
        "ConvNPFXL_NllLNPF": r"ConvNP $\mathcal{L}_{ML}$",  # for paper
        "ConvNPFL_NllLNPF": r"ConvNP $\mathcal{L}_{ML}$",  # for paper
        "ConvNPFXL_ElboLNPF": r"ConvNP $\mathcal{L}_{NP}$",  # for paper
        "SelfAttnNPF_NllLNPF": r"ANP $\mathcal{L}_{ML}$",  # for paper
        "SelfAttnNPF_ElboLNPF": r"ANP $\mathcal{L}_{NP}$",  # for paper
    },
    subtring_replace={
        "_": " ",
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
    to_upper=["Mnist", "Svhn"],
)


def prettify(table):
    """Make the name and values in a dataframe prettier / human readable."""
    renamer = lambda x: PRETTY_RENAMER[x]
    table = table.rename(columns=renamer)
    table = table.rename(index=renamer)
    table = table.applymap(renamer)
    return table


def merge_dicts(dicts):
    out = dict()
    for d in dicts:
        out.update(d)
    return out


def get_datasets_images(
    names=["mnist", "celeba64", "celeba32", "svhn", "zsmm", "zsmmt"]
):
    with silent():
        train_datasets_32 = {}
        train_datasets_64 = {}
        test_datasets_32 = {}
        test_datasets_64 = {}

        for d32 in ["mnist", "celeba32", "svhn"]:
            if d32 in names:
                (
                    train_datasets_32[d32],
                    test_datasets_32[d32],
                ) = get_train_test_img_dataset(d32)

        for d64 in ["celeba64", "zsmm", "zsmmt", "zsmms"]:
            if d64 in names:
                (
                    train_datasets_64[d64],
                    test_datasets_64[d64],
                ) = get_train_test_img_dataset(d64)

        test_datasets = merge_dicts([test_datasets_32, test_datasets_64])
        train_datasets = merge_dicts([train_datasets_32, train_datasets_64])

        train_datasets_persize = {32: train_datasets_32, 64: train_datasets_64}
        test_datasets_persize = {32: test_datasets_32, 64: test_datasets_64}

    return train_datasets, test_datasets, train_datasets_persize, test_datasets_persize


def add_y_dim(models, datasets):
    """Add y _dim to all ofthe models depending on the dataset."""
    return {
        data_name: {
            model_name: partial(model, y_dim=data_train.shape[0])
            for model_name, model in models.items()
        }
        for data_name, data_train in datasets.items()
    }


def get_exp_args(exp, is_load=True, n_runs=None, is_reeval=False):
    global_kwargs = f"--chckpnt-dirname results/neurips/{exp}/ --max-epochs 50"
    convxl_kwargs = "--n-blocks 9 --init-kernel-size 11 --n-conv-layers 2"
    convL_kwargs = (
        "--n-blocks 13 --init-kernel-size 7 --kernel-size 3 --n-conv-layers 2"
    )
    convxxl_kwargs = (
        "--n-blocks 18 --init-kernel-size 7 --kernel-size 3 --n-conv-layers 2"
    )
    convS_kwargs = "--n-blocks 4 --init-kernel-size 7 --kernel-size 3 --n-conv-layers 1"

    if is_reeval:
        global_kwargs += " --is-reeval"

    if exp == "exp_losses":
        if n_runs is None:
            n_runs = 5
        global_kwargs += " --n-z-samples 16 --batch-size 32"

        kwargs = [
            ("ConvNPFXL", "ConvNPF", f"{convxl_kwargs}",),
            ("SelfAttnNPF", "SelfAttnNPF", ""),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}  {other} {global_kwargs}"
            for loss in ["NllLNPF", "IsLNPF", "ElboLNPF"]
            for data in ["mnist"]
            for name, model, other in kwargs
        ]

    if exp == "exp_mini":
        if n_runs is None:
            n_runs = 1
        global_kwargs += " --n-z-samples 2 --batch-size 128"

        kwargs = [
            ("ConvNPFS", "ConvNPF", f"{convS_kwargs}",),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}  {other} {global_kwargs}"
            for loss in ["NllLNPF"]
            for data in ["mnist"]
            for name, model, other in kwargs
        ]

    elif exp == "exp_final_allmm":
        if n_runs is None:
            n_runs = 5
        global_kwargs += " --n-z-samples 16 --is-balanced --is-circular-padding --min-sigma 0.001 --min-lat 0.001"

        kwargs = [
            ("ConvNPFL", "ConvNPF", f"{convxl_kwargs} --batch-size 32"),
            ("SelfAttnNPF", "SelfAttnNPF", "--batch-size 8 --n-z-samples 16",),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss} {other} {global_kwargs}"
            for data in ["zsmms"]
            for loss in ["NllLNPF", "ElboLNPF"]
            for name, model, other in kwargs
        ]

    elif exp == "exp_final2_allmm":
        if n_runs is None:
            n_runs = 5
        global_kwargs += " --n-z-samples 16 --is-balanced --is-circular-padding --min-sigma 0.01 --min-lat 0.01"

        kwargs = [
            ("ConvNPFL", "ConvNPF", f"{convxl_kwargs} --batch-size 32"),
            ("SelfAttnNPF", "SelfAttnNPF", "--batch-size 8 --n-z-samples 16",),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss} {other} {global_kwargs}"
            for data in ["zsmms"]
            for loss in ["NllLNPF", "ElboLNPF"]
            for name, model, other in kwargs
        ]

    elif "exp_final3_allmm" in exp:
        if n_runs is None:
            n_runs = 5
        global_kwargs += " --n-z-samples 16 --is-balanced --n-z-samples-test 128 --is-circular-padding --min-sigma 0.001 --min-lat 0.001 --is-upper-lat --is-grad-clip"

        kwargs = [
            ("ConvNPFL", "ConvNPF", f"{convxl_kwargs} --batch-size 32"),
            ("SelfAttnNPF", "SelfAttnNPF", "--batch-size 8",),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss} {other} {global_kwargs} "
            + ("--is-iw-eval" if loss == "ElboLNPF" else "")
            for data in ["zsmms"]
            for loss in ["NllLNPF", "ElboLNPF"]
            for name, model, other in kwargs
        ]

    elif "exp_final3_allmm_hetero" in exp:
        if n_runs is None:
            n_runs = 5
        global_kwargs += " --n-z-samples 16 --is-balanced --n-z-samples-test 128 --is-circular-padding --min-sigma 0.001 --min-lat 0.001 --is-upper-lat --is-grad-clip  --is-heteroskedastic"

        kwargs = [
            ("ConvNPFL", "ConvNPF", f"{convxl_kwargs} --batch-size 32"),
            ("SelfAttnNPF", "SelfAttnNPF", "--batch-size 8",),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss} {other} {global_kwargs} "
            + ("--is-iw-eval" if loss == "ElboLNPF" else "")
            for data in ["zsmms"]
            for loss in ["NllLNPF", "ElboLNPF"]
            for name, model, other in kwargs
        ]

    elif "exp_final4_allmm_hetero" in exp:
        if n_runs is None:
            n_runs = 5
        global_kwargs += " --n-z-samples 16 --n-z-samples-test 128 --is-circular-padding --min-sigma 0.001 --min-lat 0.001 --is-upper-lat --is-grad-clip  --is-heteroskedastic"

        kwargs = [
            ("ConvNPFL", "ConvNPF", f"{convL_kwargs} --batch-size 16"),
            # ("SelfAttnNPF", "SelfAttnNPF", "--batch-size 8",),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss} {other} {global_kwargs} "
            + ("--is-iw-eval" if loss == "ElboLNPF" else "")
            for data in ["zsmms"]
            for loss in ["NllLNPF", "ElboLNPF"]
            for name, model, other in kwargs
        ]

    elif "exp_allmm_hetero_lb001_b" in exp:
        if n_runs is None:
            n_runs = 5
        global_kwargs += " --n-z-samples 32 --n-z-samples-test 64 --is-circular-padding --min-sigma 0.01 --min-lat 0.001 --is-upper-lat --is-grad-clip  --is-heteroskedastic"

        kwargs = [
            ("ConvNPFXL", "ConvNPF", f"{convxl_kwargs} --batch-size 16 --is-balanced"),
            ("ConvNPFL", "ConvNPF", f"{convL_kwargs} --batch-size 16"),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss} {other} {global_kwargs} "
            + ("--is-iw-eval" if loss == "ElboLNPF" else "")
            for data in ["zsmms"]
            for loss in ["NllLNPF", "ElboLNPF"]
            for name, model, other in kwargs
        ]

    elif "exp_allmm_hetero_lb01_b" in exp:
        if n_runs is None:
            n_runs = 5
        global_kwargs += " --n-z-samples 32 --n-z-samples-test 64 --is-circular-padding --min-sigma 0.1 --min-lat 0.001 --is-upper-lat --is-grad-clip  --is-heteroskedastic"

        kwargs = [
            ("ConvNPFXL", "ConvNPF", f"{convxl_kwargs} --batch-size 16 --is-balanced"),
            ("ConvNPFL", "ConvNPF", f"{convL_kwargs} --batch-size 16"),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss} {other} {global_kwargs} "
            + ("--is-iw-eval" if loss == "ElboLNPF" else "")
            for data in ["zsmms"]
            for loss in ["NllLNPF", "ElboLNPF"]
            for name, model, other in kwargs
        ]

    elif "exp_allmm_hetero_lb001" in exp:
        if n_runs is None:
            n_runs = 5
        global_kwargs += " --n-z-samples 32 --n-z-samples-test 64 --is-circular-padding --min-sigma 0.01 --min-lat 0.001 --is-upper-lat --is-grad-clip  --is-heteroskedastic"

        kwargs = [
            # ("ConvNPFL", "ConvNPF", f"{convxl_kwargs} --batch-size 16 --is-balanced"),
            ("ConvNPFL", "ConvNPF", f"{convL_kwargs} --batch-size 16"),
            ("SelfAttnNPF", "SelfAttnNPF", "--batch-size 8",),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss} {other} {global_kwargs} "
            + ("--is-iw-eval" if loss == "ElboLNPF" else "")
            for data in ["zsmms"]
            for loss in ["NllLNPF", "ElboLNPF"]
            for name, model, other in kwargs
        ]

    elif "exp_allmm_hetero_lb01" in exp:
        if n_runs is None:
            n_runs = 5
        global_kwargs += " --n-z-samples 32 --n-z-samples-test 64 --is-circular-padding --min-sigma 0.1 --min-lat 0.001 --is-upper-lat --is-grad-clip  --is-heteroskedastic"

        kwargs = [
            ("ConvNPFL", "ConvNPF", f"{convxl_kwargs} --batch-size 16 --is-balanced"),
            # ("ConvNPFL", "ConvNPF", f"{convL_kwargs} --batch-size 16"),
            ("SelfAttnNPF", "SelfAttnNPF", "--batch-size 8",),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss} {other} {global_kwargs} "
            + ("--is-iw-eval" if loss == "ElboLNPF" else "")
            for data in ["zsmms"]
            for loss in ["NllLNPF", "ElboLNPF"]
            for name, model, other in kwargs
        ]

    elif exp == "exp_final_losses":
        if n_runs is None:
            n_runs = 5
        global_kwargs += " --n-z-samples 16 --batch-size 16 --is-balanced --is-global --min-sigma 0.001 --min-lat 0.001 "

        kwargs = [
            ("ConvNPFXL", "ConvNPF", f"{convL_kwargs}",),
            # ("SelfAttnNPF", "SelfAttnNPF", ""),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}  {other} {global_kwargs}"
            for loss in ["NllLNPF", "ElboLNPF"]
            for data in ["mnist", "svhn", "celeba32"]
            for name, model, other in kwargs
        ]

    elif exp == "exp_final2_losses":
        if n_runs is None:
            n_runs = 5
        global_kwargs += " --n-z-samples 16 --batch-size 16 --is-balanced --is-global --min-sigma 0.01 --min-lat 0.01 "

        kwargs = [
            ("ConvNPFXL", "ConvNPF", f"{convxl_kwargs}",),
            ("SelfAttnNPF", "SelfAttnNPF", ""),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}  {other} {global_kwargs}"
            for loss in ["NllLNPF", "ElboLNPF"]
            for data in ["mnist", "svhn", "celeba32"]
            for name, model, other in kwargs
        ]

    elif "exp_final3_losses" in exp:
        if n_runs is None:
            n_runs = 5
        global_kwargs += " --n-z-samples 16 --batch-size 16 --is-balanced --is-global --min-sigma 0.001 --min-lat 0.001 --is-upper-lat --is-grad-clip"

        kwargs = [
            ("ConvNPFXL", "ConvNPF", f"{convxl_kwargs}",),
            ("SelfAttnNPF", "SelfAttnNPF", ""),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}  {other} {global_kwargs}"
            for loss in ["NllLNPF", "ElboLNPF"]
            for data in ["mnist", "svhn", "celeba32"]
            for name, model, other in kwargs
        ]

    elif "exp_losses_hetero" in exp:
        if n_runs is None:
            n_runs = 5
        global_kwargs += " --n-z-samples 16 --batch-size 16 --n-z-samples-test 128 --is-balanced --is-global --min-sigma 0.001 --min-lat 0.001 --is-upper-lat --is-grad-clip --is-heteroskedastic "

        kwargs = [
            ("ConvNPFXL", "ConvNPF", f"{convxl_kwargs}",),
            ("SelfAttnNPF", "SelfAttnNPF", ""),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}  {other} {global_kwargs}"
            + ("--is-iw-eval" if loss == "ElboLNPF" else "")  # elbo eval with iw
            for loss in ["NllLNPF", "ElboLNPF"]
            for data in ["mnist", "svhn", "celeba32"]
            for name, model, other in kwargs
        ]

    elif "exp_hetero_lb01" in exp:
        if n_runs is None:
            n_runs = 5
        global_kwargs += " --n-z-samples 16 --batch-size 16 --n-z-samples-test 128 --is-global --min-sigma 0.1 --min-lat 0.001 --is-upper-lat --is-grad-clip --is-heteroskedastic "

        kwargs = [
            ("ConvNPFXL", "ConvNPF", f"{convxl_kwargs} --batch-size 16 --is-balanced"),
            ("ConvNPFL", "ConvNPF", f"{convL_kwargs} --batch-size 16"),
            ("SelfAttnNPF", "SelfAttnNPF", ""),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}  {other} {global_kwargs}"
            + ("--is-iw-eval" if loss == "ElboLNPF" else "")  # elbo eval with iw
            for loss in ["NllLNPF", "ElboLNPF"]
            for data in ["mnist"]
            for name, model, other in kwargs
        ]

    elif "exp_hetero_lb001" in exp:
        if n_runs is None:
            n_runs = 5
        global_kwargs += " --n-z-samples 16 --batch-size 16 --n-z-samples-test 128 --is-global --min-sigma 0.01 --min-lat 0.001 --is-upper-lat --is-grad-clip --is-heteroskedastic "

        kwargs = [
            ("ConvNPFXL", "ConvNPF", f"{convxl_kwargs} --batch-size 16 --is-balanced"),
            ("ConvNPFL", "ConvNPF", f"{convL_kwargs} --batch-size 16"),
            ("SelfAttnNPF", "SelfAttnNPF", ""),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}  {other} {global_kwargs}"
            + ("--is-iw-eval" if loss == "ElboLNPF" else "")  # elbo eval with iw
            for loss in ["NllLNPF", "ElboLNPF"]
            for data in ["mnist"]
            for name, model, other in kwargs
        ]

    elif exp == "exp_qualitative":
        if n_runs is None:
            n_runs = 1
        global_kwargs += " --n-z-samples 16 --batch-size 8 --n-z-samples-test 64 --is-global --is-global --min-sigma 0.001 --min-lat 0.001 --is-upper-lat --is-grad-clip"

        kwargs = [
            ("ConvNPFXXL", "ConvNPF", f"{convxxl_kwargs}",),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}  {other} {global_kwargs}"
            for loss in ["NllLNPF"]
            for data in ["mnist", "svhn", "celeba32", "zsmms"]
            for name, model, other in kwargs
        ]

    elif exp == "exp_final_images":
        if n_runs is None:
            n_runs = 1
        global_kwargs += " --n-z-samples 16 --batch-size 8  --is-global --min-sigma 0.001 --min-lat 0.001 --is-upper-lat --is-grad-clip"

        kwargs = [
            ("ConvNPFXXL", "ConvNPF", f"{convxxl_kwargs}",),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}  {other} {global_kwargs}"
            for loss in ["NllLNPF"]
            for data in ["mnist", "svhn", "celeba32"]
            for name, model, other in kwargs
        ]

    elif exp == "exp_final2_zsamples":
        datasets = ["mnist"]
        global_kwargs += " --batch-size 16 --n-z-samples-test 64 --is-global --min-sigma 0.01 --min-lat 0.01 --is-balanced"
        if n_runs is None:
            n_runs = 5

        kwargs = [("ConvNPFXL", "ConvNPF", "NllLNPF", f"{convxl_kwargs}",)]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}_{nz}  {other} {global_kwargs} --n-z-samples {nz}"
            for data in datasets
            for nz in [2, 4, 8, 16, 32, 64]
            for name, model, loss, other in kwargs
        ]

    elif exp == "exp_final3_zsamples":
        datasets = ["mnist"]
        global_kwargs += " --batch-size 16 --n-z-samples-test 64 --is-global --min-sigma 0.001 --is-upper-lat --min-lat 0.001 --is-balanced --is-grad-clip"
        if n_runs is None:
            n_runs = 5

        kwargs = [("ConvNPFXL", "ConvNPF", "NllLNPF", f"{convxl_kwargs}",)]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}_{nz}  {other} {global_kwargs} --n-z-samples {nz}"
            for data in datasets
            for nz in [2, 4, 8, 16, 32]
            for name, model, loss, other in kwargs
        ]

    elif exp == "exp_final_zsamples":
        datasets = ["mnist"]
        global_kwargs += " --batch-size 16 --n-z-samples-test 64 --is-global --min-sigma 0.001 --min-lat 0.001 --is-balanced"
        if n_runs is None:
            n_runs = 5

        kwargs = [("ConvNPFXL", "ConvNPF", "NllLNPF", f"{convL_kwargs}",)]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}_{nz}  {other} {global_kwargs} --n-z-samples {nz}"
            for data in datasets
            for nz in [2, 4, 8, 16, 32, 64]
            for name, model, loss, other in kwargs
        ]

    elif "exp_losses_isglobal" in exp:
        if n_runs is None:
            n_runs = 5
        global_kwargs += " --n-z-samples 16 --batch-size 32 --is-global"

        kwargs = [
            ("ConvNPFXL", "ConvNPF", f"{convxl_kwargs}",),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}  {other} {global_kwargs}"
            for loss in ["NllLNPF", "IsLNPF", "ElboLNPF"]
            for data in ["mnist"]
            for name, model, other in kwargs
        ]

    elif exp == "exp_hope":
        if n_runs is None:
            n_runs = 1
        global_kwargs += " --n-z-samples 64 --batch-size 8 --is-global --min-sigma 0.01"

        kwargs = [
            ("ConvNPFXL", "ConvNPF", f"{convxl_kwargs}",),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}  {other} {global_kwargs}"
            for loss in ["NllLNPF"]
            for data in ["mnist"]
            for name, model, other in kwargs
        ]

    elif exp == "exp_global":
        if n_runs is None:
            n_runs = 3
        global_kwargs += " --n-z-samples 16 --batch-size 32 --is-global"

        kwargs = [
            ("ConvNPFXL", "ConvNPF", f"{convxl_kwargs}",),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}  {other} {global_kwargs}"
            for loss in ["NllLNPF"]
            for data in ["mnist", "svhn", "celeba32"]
            for name, model, other in kwargs
        ]

    elif exp == "exp_collapse":
        if n_runs is None:
            n_runs = 3
        global_kwargs += " --n-z-samples 16 --batch-size 32 --is-heteroskedastic"

        kwargs = [
            ("ConvNPFXL", "ConvNPF", "NllLNPF", f"{convxl_kwargs}",),
            ("ConvNPFXL", "ConvNPF", "CNPF", f"{convxl_kwargs}",),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}  {other} {global_kwargs}"
            for data in ["mnist"]
            for name, model, loss, other in kwargs
        ]

    elif exp == "exp_images":
        if n_runs is None:
            n_runs = 1
        global_kwargs += " --n-z-samples 16"

        kwargs_64 = [("ConvNPFXL", "ConvNPF", "NllLNPF", f"{convxl_kwargs}")]
        kwargs_32 = kwargs_64 + [("SelfAttnNPF", "SelfAttnNPF", "NllLNPF", "")]

        args_32 = [
            f"{model} {loss} {data} --name {name}_{loss} {other} {global_kwargs} --batch-size 32"
            for data in ["celeba32", "svhn", "mnist"]
            for name, model, loss, other in kwargs_32
        ]

        args_64 = [
            f"{model} {loss} {data} --name {name}_{loss} {other} {global_kwargs} --batch-size 16"
            for data in ["celeba64"]
            for name, model, loss, other in kwargs_64
        ]

        args_single_run = args_32 + args_64

    elif exp == "exp_images_isglobal":
        if n_runs is None:
            n_runs = 1
        global_kwargs += " --n-z-samples 16 --isglobal"

        kwargs_64 = [("ConvNPFXL", "ConvNPF", "NllLNPF", f"{convxl_kwargs}")]
        kwargs_32 = kwargs_64 + [("SelfAttnNPF", "SelfAttnNPF", "NllLNPF", "")]

        args_32 = [
            f"{model} {loss} {data} --name {name}_{loss} {other} {global_kwargs} --batch-size 32"
            for data in ["celeba32", "svhn", "mnist"]
            for name, model, loss, other in kwargs_32
        ]

        args_64 = [
            f"{model} {loss} {data} --name {name}_{loss} {other} {global_kwargs} --batch-size 16"
            for data in ["celeba64"]
            for name, model, loss, other in kwargs_64
        ]

        args_single_run = args_32 + args_64

    elif exp == "exp_images_minsig":
        if n_runs is None:
            n_runs = 1
        global_kwargs += " --n-z-samples 32"

        kwargs_64 = [("ConvNPFXL", "ConvNPF", "NllLNPF", f"{convxl_kwargs}")]
        kwargs_32 = kwargs_64 + [("SelfAttnNPF", "SelfAttnNPF", "NllLNPF", "")]

        args_32 = [
            f"{model} {loss} {data} --name {name}_{loss} {other} {global_kwargs} --batch-size 16 --min-sigma 0.03"
            for data in ["celeba32", "svhn", "mnist"]
            for name, model, loss, other in kwargs_32
        ]

        args_64 = [
            f"{model} {loss} {data} --name {name}_{loss} {other} {global_kwargs} --batch-size 8 --min-sigma 0.03"
            for data in ["celeba64"]
            for name, model, loss, other in kwargs_64
        ]

        args_single_run = args_32 + args_64

    elif exp == "exp_loglike":
        datasets = ["svhn", "celeba32"]
        global_kwargs += " --n-z-samples 16 --batch-size 32"
        if n_runs is None:
            n_runs = 5

        kwargs = [
            ("ConvNPFXL", "ConvNPF", "NllLNPF", f"{convxl_kwargs}",),
            ("SelfAttnNPF", "SelfAttnNPF", "NllLNPF", ""),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}  {other} {global_kwargs}"
            for data in datasets
            for name, model, loss, other in kwargs
        ]

    elif exp == "exp_loglike_isglobal":
        datasets = ["svhn", "celeba32"]
        global_kwargs += " --n-z-samples 16 --batch-size 32 --is-global"
        if n_runs is None:
            n_runs = 5

        kwargs = [
            ("ConvNPFXL", "ConvNPF", "NllLNPF", f"{convxl_kwargs}",),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}  {other} {global_kwargs}"
            for data in datasets
            for name, model, loss, other in kwargs
        ]

    elif exp == "exp_equivariance":
        if n_runs is None:
            n_runs = 1
        global_kwargs += " --n-z-samples 16"

        kwargs = [
            ("ConvNPF", "ConvNPF", "CNPF", "--is-circular-padding --batch-size 64",),
            ("ConvNPF", "ConvNPF", "NllLNPF", "--is-circular-padding --batch-size 64",),
            (
                "SelfAttnNPF",
                "SelfAttnNPF",
                "NllLNPF",
                "--batch-size 8 --n-z-samples 16",
            ),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss} {other} {global_kwargs}"
            for data in ["zs-multi-mnist"]
            for name, model, loss, other in kwargs
        ]

    elif exp == "exp_zsmmt":
        if n_runs is None:
            n_runs = 1
        global_kwargs += " --n-z-samples 16"

        kwargs = [
            ("ConvNPF", "ConvNPF", "CNPF", f"{convxl_kwargs} --batch-size 64",),
            ("ConvNPF", "ConvNPF", "NllLNPF", f"{convxl_kwargs} --batch-size 32"),
            (
                "SelfAttnNPF",
                "SelfAttnNPF",
                "NllLNPF",
                "--batch-size 8 --n-z-samples 16",
            ),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss} {other} {global_kwargs}"
            for data in ["zsmmt"]
            for name, model, loss, other in kwargs
        ]

    elif exp == "exp_allmm":
        if n_runs is None:
            n_runs = 1
        global_kwargs += " --n-z-samples 16 --min-sigma 0.001 --min-lat 0.001"

        kwargs = [
            ("ConvNPFXL", "ConvNPF", "NllLNPF", f"{convxl_kwargs} --batch-size 32"),
            (
                "ConvNPFXL_global",
                "ConvNPF",
                "NllLNPF",
                f"{convxl_kwargs} --batch-size 32 --is-global",
            ),
            (
                "ConvNPFXL_circular",
                "ConvNPF",
                "NllLNPF",
                f"{convxl_kwargs} --batch-size 32 --is-circular-padding",
            ),
            ("ConvNPFXXL", "ConvNPF", "NllLNPF", f"{convxxl_kwargs} --batch-size 16"),
            (
                "SelfAttnNPF",
                "SelfAttnNPF",
                "NllLNPF",
                "--batch-size 8 --n-z-samples 16",
            ),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss} {other} {global_kwargs}"
            for data in ["zsmmt", "zsmms", "mnist"]
            for name, model, loss, other in kwargs
        ]

    elif exp == "exp_minlat":
        if n_runs is None:
            n_runs = 1
        global_kwargs += " --n-z-samples 16 --min-sigma 0.01"

        kwargs = [
            ("ConvNPFXL", "ConvNPF", "NllLNPF", f"{convxl_kwargs} --batch-size 32"),
            (
                "SelfAttnNPF",
                "SelfAttnNPF",
                "NllLNPF",
                "--batch-size 8 --n-z-samples 16",
            ),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}_{min_z} {other} {global_kwargs} --min-lat {min_z}"
            for data in ["zsmms", "mnist"]
            for name, model, loss, other in kwargs
            for min_z in [0, 0.01, 0.1]
        ]

    elif exp == "exp_minpred":
        if n_runs is None:
            n_runs = 1
        global_kwargs += " --n-z-samples 16 --min-lat 0.01"

        kwargs = [
            ("ConvNPFXL", "ConvNPF", "NllLNPF", f"{convxl_kwargs} --batch-size 32"),
            (
                "SelfAttnNPF",
                "SelfAttnNPF",
                "NllLNPF",
                "--batch-size 8 --n-z-samples 16",
            ),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}_{min_pred} {other} {global_kwargs} --min-sigma {min_pred}"
            for data in ["zsmms", "mnist"]
            for name, model, loss, other in kwargs
            for min_pred in [0.0001, 0.001, 0.01]
        ]

    elif exp == "exp_zsamples":
        datasets = ["mnist"]
        global_kwargs += " --batch-size 16 --n-z-samples-test 64"
        if n_runs is None:
            n_runs = 5

        kwargs = [("ConvNPFXL", "ConvNPF", "NllLNPF", f"{convxl_kwargs}",)]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}_{nz}  {other} {global_kwargs} --n-z-samples {nz}"
            for data in datasets
            for nz in [2, 4, 8, 16, 32]
            for name, model, loss, other in kwargs
        ]

    elif exp == "exp_zsamples_isglobal":
        datasets = ["mnist"]
        global_kwargs += " --batch-size 16 --n-z-samples-test 64 --is-global"
        if n_runs is None:
            n_runs = 5

        kwargs = [("ConvNPFXL", "ConvNPF", "NllLNPF", f"{convxl_kwargs}",)]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss}_{nz}  {other} {global_kwargs} --n-z-samples {nz}"
            for data in datasets
            for nz in [2, 4, 8, 16, 32, 64]
            for name, model, loss, other in kwargs
        ]

    elif exp == "equivariance":
        datasets = ["zs-multi-mnist"]
        global_kwargs = " --max-epochs 100"
        if n_runs is None:
            n_runs = 1

        kwargs = [
            (
                "ConvNPF",
                "ConvNPF",
                "CNPF",
                "--is-circular-padding --batch-size 64 --n-z-samples 16",
            ),
            (
                "ConvNPF",
                "ConvNPF",
                "NllLNPF",
                "--is-circular-padding --batch-size 64 --n-z-samples 16",
            ),
            (
                "SelfAttnNPF",
                "SelfAttnNPF",
                "NllLNPF",
                "--batch-size 8 --n-z-samples 16",
            ),
        ]

        args_single_run = [
            f"{model} {loss} {data} --name {name}_{loss} --chckpnt-dirname results/neurips/{exp}/ {other} {global_kwargs}"
            for data in datasets
            for name, model, loss, other in kwargs
        ]

    else:
        raise ValueError(f"Unkown experiment={exp}")

    if not is_load:
        args = []
        for arg in args_single_run:
            for r in range(n_runs):
                args.append(parse_arguments((arg + f" --starting-run {r} ").split()))
    else:
        args = [
            parse_arguments((arg + f" --run {n_runs} --is-load").split())
            for arg in args_single_run
        ]

    return args


def get_trained_models(experiments=["exp_losses"], is_single_run=True, ith=None):
    with silent():  # don't print
        trained_models = dict()
        for exp in experiments:
            all_args = get_exp_args(
                exp, is_load=True, n_runs=1 if is_single_run else None
            )

            if ith is None:
                for args in all_args:
                    trainers = main(args)
                    for name, trainer in trainers.items():
                        trained_models[exp + "/" + name] = trainer
            else:
                args = all_args[ith]
                trainers = main(args)
                for name, trainer in trainers.items():
                    trained_models[exp + "/" + name] = trainer

    print(f"Loaded {len(trained_models)} models")

    return trained_models


def mean_sem_results(experiment, is_prettify=True):
    with silent():
        results = (
            load_all_results(DIR + experiment)
            .groupby(["Data", "Model"])
            .agg(["count", "mean", "std", "sem"])
            .reset_index()
        )

        out = results.pivot_table(values="LogLike", index="Model", columns="Data")[
            ["mean", "sem"]
        ]

    if is_prettify:
        out = out
    return out


def get_efficiency(trainers):
    out = []
    for name, trainer in trainers.items():
        _, best_epoch = _best_loss(trainer)
        l = [epoch_hist["dur"] for epoch_hist in trainer.history]
        out.append(
            name.split("/")
            + [best_epoch, sum(l) / len(l), count_parameters(trainer.module_)]
        )
    return pd.DataFrame(
        out,
        columns=[
            "Data",
            "Model",
            "Runs",
            "Convergence Epoch",
            "Time Per Epoch",
            "N. Param",
        ],
    )


def get_efficiencies(trainers):
    return prettify(
        (
            get_efficiency(merge_dicts(trainers.values()))
            .groupby(["Data", "Model"])
            .median()
            .reset_index()
        )
    )


def plot_qualitative_with_kde_compare(
    model1, model2, trainers, test_datasets, **kwargs
):
    data_name = model1.split("/")[1]
    name1 = model1.split("/")[2]
    name2 = model2.split("/")[2]
    plot_qualitative_with_kde(
        [PRETTY_RENAMER[name1], trainers[model1]],
        test_datasets[data_name],
        named_trainer_compare=[PRETTY_RENAMER[name2], trainers[model2],],
        figsize=(9, 7),
        percentiles=[1, 10, 20, 30, 50, 100],
        height_ratios=[1, 5],
        is_smallest_xrange=True,
        font_size=14,
        h_pad=-2,
        seed=123,
        **kwargs,
    )


def plot_qualitative_with_kde_compare_samples(
    model1, model2, trainers, test_datasets, **kwargs
):
    data_name = model1.split("/")[1]
    name1 = model1.split("/")[2]
    name2 = model2.split("/")[2]
    plot_qualitative_with_kde(
        [PRETTY_RENAMER[name1], trainers[model1]],
        test_datasets[data_name],
        named_trainer_compare=[PRETTY_RENAMER[name2], trainers[model2],],
        figsize=(9, 13),
        percentiles=[1, 5, 7, 10, 20, 100],
        height_ratios=[1, 10],
        is_smallest_xrange=True,
        font_size=14,
        h_pad=-2,
        seed=123,
        n_samples=3,
        is_select_different=True,
        **kwargs,
    )
