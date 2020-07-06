import argparse
import os
import sys
from functools import partial
from os.path import abspath, dirname

import skorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch.callbacks import GradientNormClipping, ProgressBar

from npf import (
    AttnCNP,
    AttnLNP,
    CNPFLoss,
    ELBOLossLNPF,
    GridConvCNP,
    GridConvLNP,
    NLLLossLNPF,
)
from npf.architectures import (
    CNN,
    MLP,
    GaussianConv2d,
    ResConvBlock,
    SelfAttention,
    merge_flat_input,
)
from npf.utils.datasplit import (
    GridCntxtTrgtGetter,
    RandomMasker,
    half_masker,
    no_masker,
)
from npf.utils.helpers import (
    CircularPad2d,
    MultivariateNormalDiag,
    ProbabilityConverter,
    channels_to_2nd_dim,
    channels_to_last_dim,
    make_abs_conv,
    make_padded_conv,
)
from npf.utils.predict import GenAllAutoregressivePixel
from utils.data import get_img_size, get_train_test_img_dataset
from utils.data.dataloader import cntxt_trgt_collate
from utils.train import train_models


def _update_dict(d, update):
    """Update a dictionary not in place."""
    d = d.copy()
    d.update(update)
    return d


def add_y_dim(models, datasets):
    """Add y_dim to all of the models depending on the dataset."""
    return {
        data_name: {
            model_name: partial(model, y_dim=data_train.shape[0])
            for model_name, model in models.items()
        }
        for data_name, data_train in datasets.items()
    }


def _get_train_kwargs(model_name, test_upscale_factor=1, **kwargs):
    """Return the model specific kwargs."""
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    get_cntxt_trgt = GridCntxtTrgtGetter(
        context_masker=RandomMasker(min_nnz=0.0, max_nnz=0.5),
        target_masker=no_masker,
        is_add_cntxts_to_trgts=False,
        test_upscale_factor=test_upscale_factor,
    )

    dflt_collate = cntxt_trgt_collate(get_cntxt_trgt)
    masked_collate = cntxt_trgt_collate(get_cntxt_trgt, is_return_masks=True)

    if "Attn" in model_name:
        dflt_kwargs = dict(
            iterator_train__collate_fn=dflt_collate,
            iterator_valid__collate_fn=dflt_collate,
        )

    elif "Conv" in model_name:
        dflt_kwargs = dict(
            iterator_train__collate_fn=masked_collate,
            iterator_valid__collate_fn=masked_collate,
        )

    dflt_kwargs.update(kwargs)
    return dflt_kwargs


def get_model(
    model_name,
    loss_name,
    n_blocks=5,
    init_kernel_size=None,
    kernel_size=None,
    img_shape=(32, 32),
    is_circular_padding=False,
    n_conv_layers=1,
    n_z_samples=32,
    min_sigma=0.01,
    min_lat=None,
    is_both_paths=False,
    is_heteroskedastic=False,
    is_global=False,
    n_z_samples_test=None,
    is_inbalanced=True,
    is_upper_lat=False,
    is_iw_eval=False,
):
    """Return the correct model."""

    # PARAMETERS
    neuralproc_kwargs = dict(
        r_dim=128,
        # make sure output is in 0,1 as images preprocessed so
        p_y_loc_transformer=lambda mu: torch.sigmoid(mu),
        p_y_scale_transformer=lambda y_scale: min_sigma
        + (1 - min_sigma) * F.softplus(y_scale),
    )

    if loss_name == "CNPF":
        AttnNPF = AttnCNP
        ConvNPF = GridConvCNP

    elif "LNPF" in loss_name:
        is_q_zCct = (loss_name in ["IsLNPF", "ElboLNPF"]) or is_iw_eval
        AttnNPF = partial(AttnLNP, is_q_zCct=is_q_zCct)
        ConvNPF = partial(
            GridConvLNP,
            is_q_zCct=is_q_zCct,
            is_global=is_global,
            encoded_path="both" if is_both_paths else "latent",
        )

        if n_z_samples_test is None:
            n_z_samples_test = n_z_samples * 2

        neuralproc_kwargs["n_z_samples_train"] = n_z_samples
        neuralproc_kwargs["n_z_samples_test"] = n_z_samples_test
        neuralproc_kwargs["is_heteroskedastic"] = is_heteroskedastic

        if min_lat is not None:
            if is_upper_lat:
                neuralproc_kwargs["q_z_scale_transformer"] = lambda z_scale: min_lat + (
                    1 - min_lat
                ) * torch.sigmoid(
                    z_scale - 2
                )  # start small
            else:
                neuralproc_kwargs["q_z_scale_transformer"] = lambda z_scale: min_lat + (
                    1 - min_lat
                ) * F.softplus(z_scale)

    else:
        raise ValueError(f"Unkown loss_name={loss_name}.")

    if model_name == "SelfAttnNPF":
        if not is_heteroskedastic:
            neuralproc_kwargs["XEncoder"] = partial(
                MLP, n_hidden_layers=1
            )  # historical reasons

        Model = partial(
            AttnNPF,
            x_dim=2,
            attention="transformer",
            is_self_attn=True,
            **neuralproc_kwargs,
        )

    elif model_name == "ConvNPF":
        denom = 10
        dflt_kernel_size = img_shape[-1] // denom  # currently assumes square images
        if dflt_kernel_size % 2 == 0:
            dflt_kernel_size -= 1  # make sure odd

        if init_kernel_size is None:
            init_kernel_size = dflt_kernel_size + 4
        if kernel_size is None:
            kernel_size = dflt_kernel_size

        Padder = CircularPad2d if is_circular_padding else None
        SetConv = lambda y_dim: make_padded_conv(make_abs_conv(nn.Conv2d), Padder)(
            y_dim,
            y_dim,
            groups=y_dim,
            kernel_size=init_kernel_size,
            padding=init_kernel_size // 2,
            bias=False,
        )

        if "LNPF" in loss_name:
            n_blocks_post = int(n_blocks // 3) if is_inbalanced else n_blocks + 2
            n_blocks_pre = (int(2 * n_blocks // 3) + 2) if is_inbalanced else n_blocks

            neuralproc_kwargs["CNNPostZ"] = partial(
                CNN,
                ConvBlock=ResConvBlock,
                Conv=make_padded_conv(nn.Conv2d, Padder),
                is_chan_last=True,
                kernel_size=3,  # never use a large kernel size post sampling because memory ++
                n_blocks=n_blocks_post,  # use one third after sampling because more inefficient there
                n_conv_layers=n_conv_layers,
            )
            n_blocks = n_blocks_pre  # add 2 to balance the fact that using a smaller decoder (i.e. have same # param as ConvCNP)

        Model = partial(
            ConvNPF,
            x_dim=1,  # for gridded conv it's the mask shape
            Conv=SetConv,
            CNN=partial(
                CNN,
                ConvBlock=ResConvBlock,
                Conv=make_padded_conv(nn.Conv2d, Padder),
                is_chan_last=True,
                kernel_size=kernel_size,
                n_blocks=n_blocks,
                n_conv_layers=n_conv_layers,
            ),
            **neuralproc_kwargs,
        )
    else:
        raise ValueError(f"Unkown model_name={model_name}")

    return Model


def get_Loss(loss_name):
    if loss_name == "CNPF":
        return CNPFLoss
    elif loss_name in ["NllLNPF", "IsLNPF"]:
        return NLLLossLNPF
    elif loss_name == "ElboLNPF":
        return ELBOLossLNPF
    else:
        raise ValueError(f"Unkown loss_name={loss_name}.")


def train(models, train_datasets, Loss, is_retrain=True, **kwargs):
    """Train the model."""
    return train_models(
        train_datasets,
        add_y_dim(models, train_datasets),
        Loss,
        is_retrain=is_retrain,
        train_split=skorch.dataset.CVSplit(0.1),  # use 10% of data for validation
        seed=123,
        **kwargs,
    )


def main(args):

    # DATA
    train_dataset, test_dataset = get_train_test_img_dataset(args.dataset)

    Model = get_model(
        args.model,
        args.loss,
        n_blocks=args.n_blocks,
        init_kernel_size=args.init_kernel_size,
        kernel_size=args.kernel_size,
        img_shape=get_img_size(args.dataset),
        is_circular_padding=args.is_circular_padding,
        n_conv_layers=args.n_conv_layers,
        n_z_samples=args.n_z_samples,
        min_sigma=args.min_sigma,
        min_lat=args.min_lat,
        is_both_paths=args.is_both_paths,
        is_heteroskedastic=args.is_heteroskedastic,
        is_global=args.is_global,
        n_z_samples_test=args.n_z_samples_test,
        is_inbalanced=not args.is_balanced,
        is_upper_lat=args.is_upper_lat,
        is_iw_eval=args.is_iw_eval,
    )

    model_kwargs = _get_train_kwargs(
        args.model,
        lr=args.lr,
        batch_size=args.batch_size,
        iterator_valid__batch_size=args.batch_size,
        test_upscale_factor=test_dataset.shape[1] / train_dataset.shape[1],
    )

    callbacks = []

    if args.is_progressbar:
        callbacks += [ProgressBar()]

    if args.is_grad_clip:
        callbacks += [GradientNormClipping(gradient_clip_value=1)]

    # TRAINING
    trainer = train(
        {args.name: Model},
        {args.dataset: train_dataset},
        partial(get_Loss(args.loss), is_force_mle_eval=not args.is_iw_eval),
        test_datasets={args.dataset: test_dataset},
        models_kwargs={args.name: model_kwargs},
        callbacks=callbacks,
        runs=args.runs,
        starting_run=args.starting_run,
        max_epochs=args.max_epochs,
        is_continue_train=args.is_continue_train,
        patience=args.patience,
        chckpnt_dirname=args.chckpnt_dirname,
        device=args.device,
        decay_lr=args.decay_lr,
        is_retrain=not (args.is_load or args.is_reeval),
        is_reeval=args.is_reeval,
    )

    if args.is_load:
        return trainer


def parse_arguments(args_to_parse):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", type=str, help="Model.", choices=["SelfAttnNPF", "ConvNPF"],
    )
    parser.add_argument(
        "loss",
        type=str,
        help="Loss.",
        choices=["CNPF", "NllLNPF", "IsLNPF", "ElboLNPF"],
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset.",
        choices=[
            "celeba32",
            "celeba64",
            "svhn",
            "mnist",
            "zs-multi-mnist",
            "zsmmt",
            "zsmms",
            "celeba",
            "zs-mnist",
        ],
    )

    # General optional args
    general = parser.add_argument_group("General Options")
    general.add_argument(
        "--name",
        type=str,
        help="Name of the model for saving. By default parameter from `--model`.",
    )
    general.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    general.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    general.add_argument(
        "--max-epochs", default=100, type=int, help="Max number of epochs."
    )
    general.add_argument("--runs", default=1, type=int, help="Number of runs.")
    general.add_argument(
        "--starting-run",
        default=0,
        type=int,
        help="Starting run. This is useful if a couple of r duns have already been trained, and you want to continue from there.",
    )
    general.add_argument(
        "--min-sigma",
        default=0.1,
        type=float,
        help="Lowest bound on the std that the model can predict.",
    )
    general.add_argument(
        "--min-lat",
        default=None,
        type=float,
        help="Lowest bound on the std of the variance.",
    )
    general.add_argument(
        "--is-upper-lat", action="store_true", help="Upper bound on std lat.",
    )
    general.add_argument(
        "--chckpnt-dirname",
        default="results/neurips_imgs/",
        type=str,
        help="Checkpoint and result directory.",
    )
    general.add_argument(
        "--patience", default=10, type=int, help="Patience for early stopping."
    )
    general.add_argument(
        "--decay-lr",
        default=None,
        type=int,
        help="By how much to decay the learning rate.",
    )
    general.add_argument(
        "--is-progressbar", action="store_true", help="Whether to use a progressbar."
    )
    general.add_argument("--is-grad-clip", action="store_true", help="Clip gradients.")
    general.add_argument(
        "--is-continue-train",
        action="store_true",
        help="Whether to continue training from the last checkpoint of the previous run.",
    )
    general.add_argument(
        "--n-z-samples",
        type=int,
        default=32,
        help="Number of samples to use during training.",
    )
    general.add_argument(
        "--n-z-samples-test",
        type=int,
        default=None,
        help="Number of test samples. By default uses 2x train ones.",
    )
    general.add_argument(
        "--device", type=str, default=None, help="Device.",
    )
    general.add_argument(
        "--is-heteroskedastic",
        action="store_true",
        help="Whether the *LNPs should be heteroskedastic. If using in conjuction to `NllLNPF`, it might revert to a *CNP model (collapse of latents).",
    )
    general.add_argument(
        "--is-load", action="store_true", help="Whether to load model.",
    )
    general.add_argument(
        "--is-reeval", action="store_true", help="Whether to nly reevaluate model.",
    )
    general.add_argument(
        "--is-iw-eval", action="store_true", help="Eval with IW.",
    )
    # ConvFNP options
    convfnp = parser.add_argument_group("ConvFNP Options")
    convfnp.add_argument(
        "--n-blocks",
        default=5,
        type=int,
        help="Number of blocks to use in the CNN. If using a LNPF then will use (n-blocks //2)+1 blocks before and after sampling Z.",
    )
    convfnp.add_argument(
        "--init-kernel-size", type=int, help="Kernel size to use for the set cnn."
    )
    convfnp.add_argument(
        "--kernel-size", type=int, help="Kernel size to use for the whole CNN."
    )
    convfnp.add_argument(
        "--is-circular-padding",
        action="store_true",
        help="Whether to use reflect padding.",
    )
    convfnp.add_argument(
        "--n-conv-layers",
        default=1,
        type=int,
        choices=[1, 2],
        help="How many convolutional layers to use per block.",
    )
    convfnp.add_argument(
        "--is-global",
        action="store_true",
        help="Whether to use a global latent variable in addition to the local ones.",
    )
    convfnp.add_argument(
        "--is-both-paths",
        action="store_true",
        help="Whether to use both parts det and lat.",
    )
    convfnp.add_argument(
        "--is-balanced", action="store_true", help="Whether to use balanced blocks.",
    )

    args = parser.parse_args(args_to_parse)

    if args.name is None:
        args.name = args.model

    return args


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args)
