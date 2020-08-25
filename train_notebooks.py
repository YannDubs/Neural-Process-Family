import logging
import os
import warnings
from functools import partial

import matplotlib.pyplot as plt
import skorch
import submitit
import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch.callbacks import GradientNormClipping, ProgressBar

from npf import (
    CNP,
    LNP,
    AttnCNP,
    AttnLNP,
    CNPFLoss,
    ConvCNP,
    ConvLNP,
    ELBOLossLNPF,
    GridConvCNP,
    GridConvLNP,
    NLLLossLNPF,
)
from npf.architectures import (
    CNN,
    MLP,
    ResConvBlock,
    SetConv,
    discard_ith_arg,
    merge_flat_input,
)
from npf.utils.datasplit import (
    CntxtTrgtGetter,
    GetRandomIndcs,
    GridCntxtTrgtGetter,
    RandomMasker,
    get_all_indcs,
    no_masker,
)
from npf.utils.helpers import CircularPad2d, make_abs_conv, make_padded_conv
from utils.data import cntxt_trgt_collate, get_test_upscale_factor
from utils.helpers import count_parameters
from utils.ntbks_helpers import add_y_dim, get_all_gp_datasets, get_img_datasets
from utils.train import train_models


def get_processing_kwargs(is_image, min_sigma_pred=0.01, min_lat=None):
    kwargs = dict(
        p_y_scale_transformer=lambda y_scale: min_sigma_pred
        + (1 - min_sigma_pred) * F.softplus(y_scale)
    )

    if is_image:
        kwargs["p_y_loc_transformer"] = lambda mu: torch.sigmoid(mu)

    if min_lat is not None:
        kwargs["q_z_scale_transformer"] = lambda y_scale: min_lat + (
            1 - min_lat
        ) * F.softplus(y_scale)

    return kwargs


def convlnp(
    data, is_mle=True, min_sigma_pred=0.01, min_lat=None, name="ConvLNP", **kwargs
):
    is_image = not ("Kernel" in data)

    # DATASETS
    # gp
    gp_datasets, gp_test_datasets, gp_valid_datasets = get_all_gp_datasets()
    # image
    img_datasets, img_test_datasets = get_img_datasets(["zsmms", "mnist", "celeba32"])

    # CONTEXT TARGET SPLIT
    get_cntxt_trgt_1d = cntxt_trgt_collate(
        CntxtTrgtGetter(contexts_getter=GetRandomIndcs(a=0.0, b=50))
    )
    get_cntxt_trgt_2d = cntxt_trgt_collate(
        GridCntxtTrgtGetter(context_masker=RandomMasker(a=0.0, b=0.3)),
        is_return_masks=True,  # will be using grid conv CNP => can work directly with mask
    )

    R_DIM = 128
    KWARGS = dict(
        is_q_zCct=not is_mle,  # use MLE instead of ELBO
        n_z_samples_train=16 if is_mle else 1,  # going to be more expensive
        n_z_samples_test=32,
        r_dim=R_DIM,
        Decoder=discard_ith_arg(
            torch.nn.Linear, i=0
        ),  # use small decoder because already went through CNN
        **get_processing_kwargs(
            is_image, min_sigma_pred=min_sigma_pred, min_lat=min_lat
        ),
    )

    CNN_KWARGS = dict(
        ConvBlock=ResConvBlock,
        is_chan_last=True,  # all computations are done with channel last in our code
        n_conv_layers=2,
        n_blocks=4,
    )

    # 1D case
    model_1d = partial(
        ConvLNP,
        x_dim=1,
        y_dim=1,
        CNN=partial(
            CNN,
            Conv=torch.nn.Conv1d,
            Normalization=torch.nn.BatchNorm1d,
            kernel_size=19,
            **CNN_KWARGS,
        ),
        density_induced=64,  # size of discretization
        is_global=True,  # use some global representation in addition to local
        **KWARGS,
    )

    # image (2D) case
    model_2d = partial(
        GridConvLNP,
        x_dim=1,  # for gridded conv it's the mask shape
        CNN=partial(
            CNN,
            Conv=torch.nn.Conv2d,
            Normalization=torch.nn.BatchNorm2d,
            kernel_size=9,
            **CNN_KWARGS,
        ),
        is_global=True,  # use some global representation in addition to local
        **KWARGS,
    )

    # image (2D) case when multi digit

    Padder = CircularPad2d

    model_2d_extrap = partial(
        GridConvLNP,
        x_dim=1,  # for gridded conv it's the mask shape
        CNN=partial(
            CNN,
            Normalization=torch.nn.BatchNorm2d,
            Conv=torch.nn.Conv2d,
            kernel_size=9,
            **CNN_KWARGS,
        ),
        # make first layer also padded (all arguments are defaults besides `make_padded_conv` given `Padder`)
        Conv=lambda y_dim: make_padded_conv(make_abs_conv(torch.nn.Conv2d), Padder)(
            y_dim, y_dim, groups=y_dim, kernel_size=11, padding=11 // 2, bias=False,
        ),
        # no global because multiple objects
        is_global=False,
        **KWARGS,
    )

    KWARGS = dict(
        is_retrain=True,  # whether to load precomputed model or retrain
        criterion=NLLLossLNPF if is_mle else ELBOLossLNPF,
        chckpnt_dirname="results/models/",
        device=None,
        max_epochs=50 if is_image else 100,
        lr=1e-3,
        decay_lr=10,
        seed=123,
        batch_size=16,  # smaller batch because multiple samples
        callbacks=[
            GradientNormClipping(gradient_clip_value=1),
        ],  # clipping gradients improve training
        **kwargs,
    )

    # replace the zsmm model
    models_2d = add_y_dim(
        {name: model_2d}, img_datasets
    )  # y_dim (channels) depend on data
    models_extrap = add_y_dim(
        {name: model_2d_extrap}, img_datasets
    )  # y_dim (channels) depend on data
    models_2d["zsmms"] = models_extrap["zsmms"]

    # 1D
    if not is_image:
        trainers_1d = train_models(
            {data: gp_datasets[data]},
            {name: model_1d},
            test_datasets=gp_test_datasets,
            train_split=None,
            iterator_train__collate_fn=get_cntxt_trgt_1d,
            iterator_valid__collate_fn=get_cntxt_trgt_1d,
            **KWARGS,
        )
    else:
        # 2D
        trainers_2d = train_models(
            {data: img_datasets[data]},
            models_2d,
            test_datasets=img_test_datasets,
            train_split=skorch.dataset.CVSplit(
                0.1
            ),  # use 10% of training for valdiation
            iterator_train__collate_fn=get_cntxt_trgt_2d,
            iterator_valid__collate_fn=get_cntxt_trgt_2d,
            **KWARGS,
        )


def convlnp2(
    data, is_mle=True, min_sigma_pred=0.01, min_lat=None, name="ConvLNP2", **kwargs
):
    is_image = not ("Kernel" in data)

    # DATASETS
    # gp
    gp_datasets, gp_test_datasets, gp_valid_datasets = get_all_gp_datasets()
    # image
    img_datasets, img_test_datasets = get_img_datasets(["zsmms", "mnist", "celeba32"])

    # CONTEXT TARGET SPLIT
    get_cntxt_trgt_1d = cntxt_trgt_collate(
        CntxtTrgtGetter(contexts_getter=GetRandomIndcs(a=0.0, b=50))
    )
    get_cntxt_trgt_2d = cntxt_trgt_collate(
        GridCntxtTrgtGetter(
            context_masker=RandomMasker(a=0.0, b=0.1, proba_uniform=0.1)
        ),
        is_return_masks=True,  # will be using grid conv CNP => can work directly with mask
    )

    R_DIM = 128
    KWARGS = dict(
        is_q_zCct=not is_mle,  # use MLE instead of ELBO
        n_z_samples_train=16 if is_mle else 1,  # going to be more expensive
        n_z_samples_test=32,
        r_dim=R_DIM,
        Decoder=discard_ith_arg(
            torch.nn.Linear, i=0
        ),  # use small decoder because already went through CNN
        **get_processing_kwargs(
            is_image, min_sigma_pred=min_sigma_pred, min_lat=min_lat
        ),
    )

    CNN_KWARGS = dict(
        ConvBlock=ResConvBlock,
        is_chan_last=True,  # all computations are done with channel last in our code
        n_conv_layers=2,
        n_blocks=4,
    )

    # 1D case
    model_1d = partial(
        ConvLNP,
        x_dim=1,
        y_dim=1,
        CNN=partial(
            CNN,
            Conv=torch.nn.Conv1d,
            Normalization=torch.nn.BatchNorm1d,
            kernel_size=19,
            **CNN_KWARGS,
        ),
        density_induced=64,  # size of discretization
        is_global=True,  # use some global representation in addition to local
        **KWARGS,
    )

    # image (2D) case
    model_2d = partial(
        GridConvLNP,
        x_dim=1,  # for gridded conv it's the mask shape
        CNN=partial(
            CNN,
            Conv=torch.nn.Conv2d,
            Normalization=torch.nn.BatchNorm2d,
            kernel_size=9,
            **CNN_KWARGS,
        ),
        is_global=True,  # use some global representation in addition to local
        **KWARGS,
    )

    # image (2D) case when multi digit

    Padder = CircularPad2d

    model_2d_extrap = partial(
        GridConvLNP,
        x_dim=1,  # for gridded conv it's the mask shape
        CNN=partial(
            CNN,
            Normalization=torch.nn.BatchNorm2d,
            Conv=torch.nn.Conv2d,
            kernel_size=9,
            **CNN_KWARGS,
        ),
        # make first layer also padded (all arguments are defaults besides `make_padded_conv` given `Padder`)
        Conv=lambda y_dim: make_padded_conv(make_abs_conv(torch.nn.Conv2d), Padder)(
            y_dim, y_dim, groups=y_dim, kernel_size=11, padding=11 // 2, bias=False,
        ),
        # no global because multiple objects
        is_global=False,
        **KWARGS,
    )

    KWARGS = dict(
        is_retrain=True,  # whether to load precomputed model or retrain
        criterion=NLLLossLNPF if is_mle else ELBOLossLNPF,
        chckpnt_dirname="results/models/",
        device=None,
        max_epochs=50 if is_image else 100,
        lr=1e-3,
        decay_lr=10,
        seed=123,
        batch_size=16,  # smaller batch because multiple samples
        callbacks=[
            GradientNormClipping(gradient_clip_value=1),
        ],  # clipping gradients improve training
        **kwargs,
    )

    # replace the zsmm model
    models_2d = add_y_dim(
        {name: model_2d}, img_datasets
    )  # y_dim (channels) depend on data
    models_extrap = add_y_dim(
        {name: model_2d_extrap}, img_datasets
    )  # y_dim (channels) depend on data
    models_2d["zsmms"] = models_extrap["zsmms"]

    # 1D
    if not is_image:
        trainers_1d = train_models(
            {data: gp_datasets[data]},
            {name: model_1d},
            test_datasets=gp_test_datasets,
            train_split=None,
            iterator_train__collate_fn=get_cntxt_trgt_1d,
            iterator_valid__collate_fn=get_cntxt_trgt_1d,
            **KWARGS,
        )
    else:
        # 2D
        trainers_2d = train_models(
            {data: img_datasets[data]},
            models_2d,
            test_datasets=img_test_datasets,
            train_split=skorch.dataset.CVSplit(
                0.1
            ),  # use 10% of training for valdiation
            iterator_train__collate_fn=get_cntxt_trgt_2d,
            iterator_valid__collate_fn=get_cntxt_trgt_2d,
            **KWARGS,
        )


def convcnp(data, min_sigma_pred=0.01, name="ConvCNP", **kwargs):
    is_image = not ("Kernel" in data)

    # DATASETS
    # gp
    gp_datasets, gp_test_datasets, gp_valid_datasets = get_all_gp_datasets()
    # image
    img_datasets, img_test_datasets = get_img_datasets(["zsmms", "mnist", "celeba32"])
    # uncomment next line if want to train on large celeba (time ++)
    imgXL_datasets, imgXL_test_datasets = get_img_datasets(["celeba128"])

    # CONTEXT TARGET SPLIT
    get_cntxt_trgt_1d = cntxt_trgt_collate(
        CntxtTrgtGetter(contexts_getter=GetRandomIndcs(a=0.0, b=50))
    )
    get_cntxt_trgt_2d = cntxt_trgt_collate(
        GridCntxtTrgtGetter(context_masker=RandomMasker(a=0.0, b=0.3)),
        is_return_masks=True,  # will be using grid conv CNP => can work directly with mask
    )
    get_cntxt_trgt_2dXL = cntxt_trgt_collate(
        GridCntxtTrgtGetter(context_masker=RandomMasker(a=0.0, b=0.05)),
        is_return_masks=True,  # use only 10% of the data because much easier task in larger images
    )
    get_cntxt_trgt_2dXL_bis = cntxt_trgt_collate(
        GridCntxtTrgtGetter(
            context_masker=RandomMasker(
                a=0.0, b=0.03, proba_uniform=0.3, is_ensure_one=True
            )
        ),
        is_return_masks=True,  # use only 10% of the data because much easier task in larger images
    )

    R_DIM = 128
    KWARGS = dict(
        r_dim=R_DIM, **get_processing_kwargs(is_image, min_sigma_pred=min_sigma_pred)
    )

    CNN_KWARGS = dict(
        ConvBlock=ResConvBlock,
        is_chan_last=True,  # all computations are done with channel last in our code
        n_conv_layers=2,
    )

    # 1D case
    model_1d = partial(
        ConvCNP,
        x_dim=1,
        y_dim=1,
        CNN=partial(
            CNN,
            Conv=torch.nn.Conv1d,
            Normalization=torch.nn.BatchNorm1d,
            n_blocks=5,
            kernel_size=19,
            **CNN_KWARGS,
        ),
        density_induced=64,  # size of discretization
        **KWARGS,
    )

    # image (2D) case
    model_2d = partial(
        GridConvCNP,
        x_dim=1,  # for gridded conv it's the mask shape
        CNN=partial(
            CNN,
            Conv=torch.nn.Conv2d,
            Normalization=torch.nn.BatchNorm2d,
            n_blocks=5,
            kernel_size=9,
            **CNN_KWARGS,
        ),
        **KWARGS,
    )

    # image (2D) case when multi digit

    Padder = CircularPad2d

    model_2d_extrap = partial(
        GridConvCNP,
        x_dim=1,  # for gridded conv it's the mask shape
        CNN=partial(
            CNN,
            Conv=make_padded_conv(torch.nn.Conv2d, Padder),
            Normalization=partial(torch.nn.BatchNorm2d, eps=1e-2),
            n_blocks=5,
            kernel_size=9,
            **CNN_KWARGS,
        ),
        # make first layer also padded (all arguments are defaults besides `make_padded_conv` given `Padder`)
        Conv=lambda y_dim: make_padded_conv(make_abs_conv(torch.nn.Conv2d), Padder)(
            y_dim, y_dim, groups=y_dim, kernel_size=11, padding=11 // 2, bias=False,
        ),
        **KWARGS,
    )

    # image (2D) case XL
    model_2d_XL = partial(
        GridConvCNP,
        x_dim=1,  # for gridded conv it's the mask shape
        CNN=partial(
            CNN,
            Conv=torch.nn.Conv2d,
            Normalization=torch.nn.BatchNorm2d,
            n_blocks=12,
            kernel_size=9,
            **CNN_KWARGS,
        ),
        **KWARGS,
    )

    model_2d_XL_bis = partial(
        GridConvCNP,
        x_dim=1,  # for gridded conv it's the mask shape
        CNN=partial(
            CNN,
            Conv=torch.nn.Conv2d,
            Normalization=torch.nn.BatchNorm2d,
            n_blocks=12,
            kernel_size=9,
            **CNN_KWARGS,
        ),
        **KWARGS,
    )

    KWARGS = dict(
        is_retrain=True,  # whether to load precomputed model or retrain
        criterion=CNPFLoss,
        chckpnt_dirname="results/models/",
        device=None,
        max_epochs=50 if is_image else 100,
        lr=1e-3,
        decay_lr=10,
        seed=123,
        batch_size=32,
        callbacks=[
            ProgressBar(),
            # GradientNormClipping(gradient_clip_value=1),
        ],  # clipping gradients improve training
        **kwargs,
    )

    # replace the zsmm model
    models_2d = add_y_dim(
        {name: model_2d}, img_datasets
    )  # y_dim (channels) depend on data
    models_extrap = add_y_dim({"ConvCNP": model_2d_extrap}, img_datasets)
    models_2d["zsmms"] = models_extrap["zsmms"]

    # 1D
    if not is_image:
        trainers_1d = train_models(
            {data: gp_datasets[data]},
            {name: model_1d},
            test_datasets=gp_test_datasets,
            train_split=None,
            iterator_train__collate_fn=get_cntxt_trgt_1d,
            iterator_valid__collate_fn=get_cntxt_trgt_1d,
            **KWARGS,
        )
    elif data == "celeba128":
        trainers_2d = train_models(
            imgXL_datasets,
            add_y_dim(
                {f"{name}XL": model_2d_XL}, imgXL_datasets
            ),  # y_dim (channels) depend on data
            test_datasets=imgXL_test_datasets,
            train_split=skorch.dataset.CVSplit(
                0.1
            ),  # use 10% of training for valdiation
            iterator_train__collate_fn=get_cntxt_trgt_2dXL,
            iterator_valid__collate_fn=get_cntxt_trgt_2dXL,
            **KWARGS,
        )
    elif data == "celeba128bis":
        trainers_2d = train_models(
            imgXL_datasets,
            add_y_dim(
                {f"{name}XL_bis": model_2d_XL_bis}, imgXL_datasets
            ),  # y_dim (channels) depend on data
            test_datasets=imgXL_test_datasets,
            train_split=skorch.dataset.CVSplit(
                0.1
            ),  # use 10% of training for valdiation
            iterator_train__collate_fn=get_cntxt_trgt_2dXL_bis,
            iterator_valid__collate_fn=get_cntxt_trgt_2dXL_bis,
            **KWARGS,
        )
    else:
        # 2D
        trainers_2d = train_models(
            {data: img_datasets[data]},
            models_2d,
            test_datasets=img_test_datasets,
            train_split=skorch.dataset.CVSplit(
                0.1
            ),  # use 10% of training for valdiation
            iterator_train__collate_fn=get_cntxt_trgt_2d,
            iterator_valid__collate_fn=get_cntxt_trgt_2d,
            **KWARGS,
        )


def attncnp(data, min_sigma_pred=0.01, name="AttnCNP", **kwargs):
    is_image = not ("Kernel" in data)

    # DATASETS
    # gp
    gp_datasets, gp_test_datasets, gp_valid_datasets = get_all_gp_datasets()
    # image
    img_datasets, img_test_datasets = get_img_datasets(["zsmms", "mnist", "celeba32"])

    # CONTEXT TARGET SPLIT
    get_cntxt_trgt_1d = cntxt_trgt_collate(
        CntxtTrgtGetter(contexts_getter=GetRandomIndcs(a=0.0, b=50))
    )
    get_cntxt_trgt_2d = cntxt_trgt_collate(
        GridCntxtTrgtGetter(context_masker=RandomMasker(a=0.0, b=0.3))
    )

    # for ZXMMS you need the pixels to not be in [-1,1] but [-1.75,1.75] (i.e 56 / 32) because you are extrapolating
    get_cntxt_trgt_2d_extrap = cntxt_trgt_collate(
        GridCntxtTrgtGetter(
            context_masker=RandomMasker(a=0, b=0.5),
            upscale_factor=get_test_upscale_factor("zsmms"),
        )
    )

    R_DIM = 128
    KWARGS = dict(
        r_dim=R_DIM,
        attention="transformer",
        **get_processing_kwargs(is_image, min_sigma_pred=min_sigma_pred),
    )

    # 1D case
    model_1d = partial(
        AttnCNP,
        x_dim=1,
        y_dim=1,
        XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
            partial(MLP, n_hidden_layers=2, hidden_size=R_DIM), is_sum_merge=True,
        ),
        is_self_attn=False,
        **KWARGS,
    )

    # image (2D) case
    model_2d = partial(
        AttnCNP, x_dim=2, is_self_attn=True, **KWARGS
    )  # don't add y_dim yet because depends on data

    KWARGS = dict(
        is_retrain=True,  # whether to load precomputed model or retrain
        criterion=CNPFLoss,
        chckpnt_dirname="results/models/",
        device=None,
        max_epochs=50 if is_image else 100,
        lr=1e-3,
        decay_lr=10,
        seed=123,
        batch_size=32,
        callbacks=[],
        **kwargs,
    )

    # 1D
    if not is_image:
        trainers_1d = train_models(
            {data: gp_datasets[data]},
            {name: model_1d},
            test_datasets=gp_test_datasets,
            train_split=None,
            iterator_train__collate_fn=get_cntxt_trgt_1d,
            iterator_valid__collate_fn=get_cntxt_trgt_1d,
            **KWARGS,
        )
    else:
        # 2D
        trainers_2d = train_models(
            {data: img_datasets[data]},
            add_y_dim(
                {name: model_2d}, img_datasets
            ),  # y_dim (channels) depend on data,
            test_datasets=img_test_datasets,
            train_split=skorch.dataset.CVSplit(
                0.1
            ),  # use 10% of training for valdiation
            iterator_train__collate_fn=get_cntxt_trgt_2d,
            iterator_valid__collate_fn=get_cntxt_trgt_2d,
            datasets_kwargs=dict(
                zsmms=dict(iterator_valid__collate_fn=get_cntxt_trgt_2d_extrap,)
            ),  # for zsmm use extrapolation
            **KWARGS,
        )


def attnlnp(
    data, is_mle=False, min_sigma_pred=0.01, min_lat=None, name="AttnLNP", **kwargs
):
    is_image = not ("Kernel" in data)

    # DATASETS
    # gp
    gp_datasets, gp_test_datasets, gp_valid_datasets = get_all_gp_datasets()
    # image
    img_datasets, img_test_datasets = get_img_datasets(["zsmms", "mnist", "celeba32"])
    # uncomment next line if want to train on large celeba (time ++)
    imgXL_datasets, imgXL_test_datasets = get_img_datasets(["celeba128"])

    # CONTEXT TARGET SPLIT
    get_cntxt_trgt_1d = cntxt_trgt_collate(
        CntxtTrgtGetter(contexts_getter=GetRandomIndcs(a=0.0, b=50))
    )
    get_cntxt_trgt_2d = cntxt_trgt_collate(
        GridCntxtTrgtGetter(context_masker=RandomMasker(a=0.0, b=0.3))
    )

    # for ZXMMS you need the pixels to not be in [-1,1] but [-1.75,1.75] (i.e 56 / 32) because you are extrapolating
    get_cntxt_trgt_2d_extrap = cntxt_trgt_collate(
        GridCntxtTrgtGetter(
            context_masker=RandomMasker(a=0, b=0.5),
            upscale_factor=get_test_upscale_factor("zsmms"),
        )
    )

    R_DIM = 128
    KWARGS = dict(
        is_q_zCct=not is_mle,  # use MLE instead of ELBO
        n_z_samples_train=8 if is_mle else 1,  # going to be more expensive
        n_z_samples_test=8,
        r_dim=R_DIM,
        attention="transformer",
        **get_processing_kwargs(
            is_image, min_sigma_pred=min_sigma_pred, min_lat=min_lat
        ),
    )

    # 1D case
    model_1d = partial(
        AttnLNP,
        x_dim=1,
        y_dim=1,
        XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
            partial(MLP, n_hidden_layers=2, hidden_size=R_DIM), is_sum_merge=True,
        ),
        is_self_attn=False,
        **KWARGS,
    )

    # image (2D) case
    model_2d = partial(
        AttnLNP, x_dim=2, is_self_attn=True, **KWARGS
    )  # don't add y_dim yet because depends on data

    KWARGS = dict(
        is_retrain=True,  # whether to load precomputed model or retrain
        criterion=NLLLossLNPF if is_mle else ELBOLossLNPF,
        chckpnt_dirname="results/models/",
        device=None,
        max_epochs=50 if is_image else 100,
        lr=1e-3,
        decay_lr=10,
        seed=123,
        batch_size=32,
        callbacks=[],
        **kwargs,
    )

    # 1D
    if not is_image:
        trainers_1d = train_models(
            {data: gp_datasets[data]},
            {name: model_1d},
            test_datasets=gp_test_datasets,
            train_split=None,
            iterator_train__collate_fn=get_cntxt_trgt_1d,
            iterator_valid__collate_fn=get_cntxt_trgt_1d,
            **KWARGS,
        )
    else:
        # 2D
        trainers_2d = train_models(
            {data: img_datasets[data]},
            add_y_dim(
                {name: model_2d}, img_datasets
            ),  # y_dim (channels) depend on data,
            test_datasets=img_test_datasets,
            train_split=skorch.dataset.CVSplit(
                0.1
            ),  # use 10% of training for valdiation
            iterator_train__collate_fn=get_cntxt_trgt_2d,
            iterator_valid__collate_fn=get_cntxt_trgt_2d,
            datasets_kwargs=dict(
                zsmms=dict(iterator_valid__collate_fn=get_cntxt_trgt_2d_extrap,)
            ),  # for zsmm use extrapolation
            **KWARGS,
        )


def cnp(data, min_sigma_pred=0.01, name="CNP", **kwargs):
    is_image = not ("Kernel" in data)

    # DATASETS
    # merges : get_datasets_single_gp, get_datasets_varying_hyp_gp, get_datasets_varying_kernel_gp
    gp_datasets, gp_test_datasets, gp_valid_datasets = get_all_gp_datasets()
    # image datasets
    img_datasets, img_test_datasets = get_img_datasets(["celeba32", "mnist", "zsmms"])

    # CONTEXT TARGET SPLIT
    get_cntxt_trgt_1d = cntxt_trgt_collate(
        CntxtTrgtGetter(
            contexts_getter=GetRandomIndcs(a=0.0, b=50), targets_getter=get_all_indcs,
        )
    )
    # same as in 1D but with masks (2d) rather than indices
    get_cntxt_trgt_2d = cntxt_trgt_collate(
        GridCntxtTrgtGetter(
            context_masker=RandomMasker(a=0.0, b=0.3), target_masker=no_masker,
        )
    )

    # for ZSMMS you need the pixels to not be in [-1,1] but [-1.75,1.75] (i.e 56 / 32) because you are extrapolating
    get_cntxt_trgt_2d_extrap = cntxt_trgt_collate(
        GridCntxtTrgtGetter(
            context_masker=RandomMasker(a=0, b=0.3, proba_uniform=0.5),
            target_masker=no_masker,
            upscale_factor=get_test_upscale_factor("zsmms"),
        )
    )

    R_DIM = 128
    KWARGS = dict(
        XEncoder=partial(MLP, n_hidden_layers=1, hidden_size=R_DIM),
        Decoder=merge_flat_input(  # MLP takes single input but we give x and R so merge them
            partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
        ),
        r_dim=R_DIM,
        **get_processing_kwargs(is_image, min_sigma_pred=min_sigma_pred),
    )

    # 1D case
    model_1d = partial(
        CNP,
        x_dim=1,
        y_dim=1,
        XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
            partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 2), is_sum_merge=True,
        ),
        **KWARGS,
    )

    # image (2D) case
    model_2d = partial(
        CNP,
        x_dim=2,
        XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
            partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 3), is_sum_merge=True,
        ),
        **KWARGS,
    )  # don't add y_dim yet because depends on data

    KWARGS = dict(
        is_retrain=True,  # whether to load precomputed model or retrain
        criterion=CNPFLoss,  # Standard loss for conditional NPFs
        chckpnt_dirname="results/models/",
        device=None,  # use GPU if available
        max_epochs=50 if is_image else 100,
        batch_size=32,
        lr=1e-3,
        decay_lr=10,  # decrease learning rate by 10 during training
        seed=123,
        callbacks=[],
        **kwargs,
    )

    # 1D
    if not is_image:
        trainers_1d = train_models(
            {data: gp_datasets[data]},
            {name: model_1d},
            test_datasets=gp_test_datasets,
            train_split=None,
            iterator_train__collate_fn=get_cntxt_trgt_1d,
            iterator_valid__collate_fn=get_cntxt_trgt_1d,
            **KWARGS,
        )
    else:
        # 2D
        trainers_2d = train_models(
            {data: img_datasets[data]},
            add_y_dim(
                {name: model_2d}, img_datasets
            ),  # y_dim (channels) depend on data,
            test_datasets=img_test_datasets,
            train_split=skorch.dataset.CVSplit(
                0.1
            ),  # use 10% of training for valdiation
            iterator_train__collate_fn=get_cntxt_trgt_2d,
            iterator_valid__collate_fn=get_cntxt_trgt_2d,
            datasets_kwargs=dict(
                zsmms=dict(iterator_valid__collate_fn=get_cntxt_trgt_2d_extrap)
            ),  # for zsmm use extrapolation
            **KWARGS,
        )


def lnp(data, is_mle=False, min_sigma_pred=0.01, min_lat=None, name="LNP", **kwargs):
    is_image = not ("Kernel" in data)

    # DATASETS
    # gp
    gp_datasets, gp_test_datasets, gp_valid_datasets = get_all_gp_datasets()
    # image
    img_datasets, img_test_datasets = get_img_datasets(["celeba32", "mnist", "zsmms"])

    # CONTEXT TARGET SPLIT
    get_cntxt_trgt_1d = cntxt_trgt_collate(
        CntxtTrgtGetter(
            contexts_getter=GetRandomIndcs(a=0.0, b=50), targets_getter=get_all_indcs,
        )
    )
    # same as in 1D but with masks (2d) rather than indices
    get_cntxt_trgt_2d = cntxt_trgt_collate(
        GridCntxtTrgtGetter(
            context_masker=RandomMasker(a=0.0, b=0.3), target_masker=no_masker,
        )
    )

    # for ZSMMS you need the pixels to not be in [-1,1] but [-1.75,1.75] (i.e 56 / 32) because you are extrapolating
    get_cntxt_trgt_2d_extrap = cntxt_trgt_collate(
        GridCntxtTrgtGetter(
            context_masker=RandomMasker(a=0, b=0.3),
            target_masker=no_masker,
            upscale_factor=get_test_upscale_factor("zsmms"),
        )
    )

    R_DIM = 128
    KWARGS = dict(
        is_q_zCct=not is_mle,  # use MLE instead of ELBO
        n_z_samples_train=32 if is_mle else 1,  # going to be more expensive
        n_z_samples_test=32,
        XEncoder=partial(MLP, n_hidden_layers=1, hidden_size=R_DIM),
        Decoder=merge_flat_input(  # MLP takes single input but we give x and R so merge them
            partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
        ),
        r_dim=R_DIM,
        **get_processing_kwargs(
            is_image, min_sigma_pred=min_sigma_pred, min_lat=min_lat
        ),
    )

    # 1D case
    model_1d = partial(
        LNP,
        x_dim=1,
        y_dim=1,
        XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
            partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 2), is_sum_merge=True,
        ),
        **KWARGS,
    )

    # image (2D) case
    model_2d = partial(
        LNP,
        x_dim=2,
        XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
            partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 3), is_sum_merge=True,
        ),
        **KWARGS,
    )  # don't add y_dim yet because depends on data

    KWARGS = dict(
        is_retrain=True,  # whether to load precomputed model or retrain
        criterion=NLLLossLNPF
        if is_mle
        else ELBOLossLNPF,  # (approx) conditional ELBO Loss
        chckpnt_dirname="results/models/",
        device=None,  # use GPU if available
        max_epochs=50 if is_image else 100,
        batch_size=32,
        lr=1e-3,
        decay_lr=10,  # decrease learning rate by 10 during training
        seed=123,
        callbacks=[],
        **kwargs,
    )

    # 1D
    if not is_image:
        trainers_1d = train_models(
            {data: gp_datasets[data]},
            {name: model_1d},
            test_datasets=gp_test_datasets,
            train_split=None,
            iterator_train__collate_fn=get_cntxt_trgt_1d,
            iterator_valid__collate_fn=get_cntxt_trgt_1d,
            **KWARGS,
        )
    else:
        # 2D
        trainers_2d = train_models(
            {data: img_datasets[data]},
            add_y_dim(
                {name: model_2d}, img_datasets
            ),  # y_dim (channels) depend on data,
            test_datasets=img_test_datasets,
            train_split=skorch.dataset.CVSplit(
                0.1
            ),  # use 10% of training for valdiation
            iterator_train__collate_fn=get_cntxt_trgt_2d,
            iterator_valid__collate_fn=get_cntxt_trgt_2d,
            datasets_kwargs=dict(
                zsmms=dict(iterator_valid__collate_fn=get_cntxt_trgt_2d_extrap)
            ),  # for zsmm use extrapolation
            **KWARGS,
        )


class Run:
    def __init__(self, notebook):
        super().__init__()
        self.notebook = notebook

    def checkpoint(self, args):
        """Resubmits the same callable with the same arguments but makes sure continnue from last chckpnt."""
        args["is_continue_train"] = True
        return submitit.helpers.DelayedSubmission(self, args)

    def __call__(self, args):
        return eval(self.notebook)(**args)


# import submitit

# from train_notebooks import Run
# from utils.train import train_models

# executor = submitit.AutoExecutor(folder="logs/%j", slurm_max_num_timeout=3)
# executor.update_parameters(
#     gpus_per_node=1,
#     cpus_per_task=10,
#     slurm_time=60 * 24 * 1,
#     slurm_mem="32GB",
#     slurm_constraint="volta32gb",
#     # slurm_partition="dev",
# )


# jobs = executor.map_array(
#     Run("cnpHope"),
#     [
#         dict(data="RBF_Kernel"),
#         dict(data="Periodic_Kernel"),
#         dict(data="Noisy_Matern_Kernel"),
#         dict(data="Variable_Matern_Kernel"),
#         dict(data="All_Kernels"),
#         dict(data="celeba32"),
#         dict(data="mnist"),
#         dict(data="zsmms"),
#     ],
# )


# jobs

# for j in jobs:
#     print(j.stderr())
