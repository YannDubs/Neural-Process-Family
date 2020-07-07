# Attentive Conditional Neural Process (AttnCNP)

```{figure} images/computational_graph_AttnCNPs.svg
---
height: 250px
name: computational_graph_AttnCNPs
---
Computational graph for Attentive Conditional Neural Processes.
```

AttnCNPs differ from other CNPFs in that they use attention {cite}`bahdanau2014neural`  for the aggregator and and usually apply a self-attention layer on the context set.


## Properties

CNPs have the following desirable properties compared to other CNPFs:

* &#10003; **Less Underfitting**. The main advantage compared to {doc}`CNPs <CNP>` is that the representation of the context set is now target specific $R^{(t)}$ (as seen in {numref}`computational_graph_AttnCNPs` compared to {numref}`computational_graph_CNPs`). This enables the model to greatly decrease underfitting. Intuitively, if a target point $t$ is very close to a context point $t$ then the representation used in the posterior predictive of $t$.

But it suffers from the following issues:

* &#10007; **$\mathbf{\mathcal{O}(C(T+C))}$ Inference**. First, a self-attention layer is applied to the context set $\mathcal{O}(C^2)$. For each target point a representation $R^{(t)}$ is then computed using cross-attention $\mathcal{O}(C*T)$. Inference is thus $\mathcal{O}(C(T+C))$ (compared to $\mathcal{O}(T+C)$ for {doc}`CNPs <CNP>`). In practive, the model is not much slower as attention can be parallelized on a GPU.

* &#10007; **Cannot extrapolate**. The predictions outside of the training range are terrible because neural networks that are very non linear and known to bad at extrapolating {cite}`dubois2019location`.



%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import logging
import os
import warnings

import matplotlib.pyplot as plt
import torch

os.chdir("..")

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
logging.disable(logging.ERROR)

N_THREADS = 8
IS_FORCE_CPU = False  # Nota Bene : notebooks don't deallocate GPU memory

if IS_FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

torch.set_num_threads(N_THREADS)

## Initialization

Let's load the {doc}`data <Datasets>` and define the context target splitter.
Here, we select uniformly between 0.0 and 0.5 context points and use all points as target. 

from npf.utils.datasplit import (
    CntxtTrgtGetter,
    GetRandomIndcs,
    GridCntxtTrgtGetter,
    RandomMasker,
)
from utils.data import cntxt_trgt_collate, get_test_upscale_factor
from utils.ntbks_helpers import get_all_gp_datasets, get_img_datasets

# DATASETS
# gp
gp_datasets, gp_test_datasets, gp_valid_datasets = get_all_gp_datasets()
# image
img_datasets, img_test_datasets = get_img_datasets(["celeba32", "mnist", "zsmms"])

# CONTEXT TARGET SPLIT
get_cntxt_trgt_1d = cntxt_trgt_collate(
    CntxtTrgtGetter(contexts_getter=GetRandomIndcs(min_n_indcs=0.0, max_n_indcs=0.5))
)
get_cntxt_trgt_2d = cntxt_trgt_collate(
    GridCntxtTrgtGetter(context_masker=RandomMasker(min_nnz=0.0, max_nnz=0.5))
)

# for ZXMMS you need the pixels to not be in [-1,1] but [-1.75,1.75] (i.e 56 / 32) because you are extrapolating
get_cntxt_trgt_2d_extrap = cntxt_trgt_collate(
    GridCntxtTrgtGetter(
        context_masker=RandomMasker(min_nnz=0.0, max_nnz=0.5),
        test_upscale_factor=get_test_upscale_factor("zsmms"),
    )
)

Let's now define the models. For both the 1D and 2D case we will be using the following:
* **Encoder** $\mathrm{e}_{\boldsymbol{\theta}}$ : a 1-hidden layer MLP that encodes the features, followed by
    * 1D : 2 hidden layer MLP that encodes each feature-value pair.
    * 2D : two self attention layers[^selfattn] each implemented as 8-headed attention, a skip connection, and two layer normalizations (as in {cite}`kim2019attentive`).
* **Aggregator** $\mathrm{Agg}$: multi-head cross-attention layer.
* **Decoder** $\mathrm{d}_{\boldsymbol{\theta}}$: a 4 hidden layer MLP that predicts the distribution of the target value given the global representation and target context.

All hidden representations will be of 128 dimensions.

For more details about all the possible parameters, refer to the docstrings of `AttnCNP` and the base class `NeuralProcessFamily`.

# AttnCNP Docstring
from npf import AttnCNP

print(AttnCNP.__doc__)

from functools import partial

from npf.architectures import MLP, merge_flat_input
from utils.helpers import count_parameters

R_DIM = 128
KWARGS = dict(r_dim=R_DIM, attention="transformer",)

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

n_params_1d = count_parameters(model_1d())
n_params_2d = count_parameters(model_2d(y_dim=3))
print(f"Number Parameters (1D): {n_params_1d:,d}")
print(f"Number Parameters (2D): {n_params_2d:,d}")

### Training

The main function for training is `train_models` which trains a dictionary of models on a dictionary of datasets and returns all the trained models.
See its docstring for possible parameters.

import skorch
from npf import CNPFLoss
from utils.ntbks_helpers import add_y_dim
from utils.train import train_models

KWARGS = dict(
    is_retrain=False,  # whether to load precomputed model or retrain
    criterion=CNPFLoss,
    chckpnt_dirname="results/npfs/ntbks/",
    device=None,
    max_epochs=50,
    lr=1e-3,
    decay_lr=10,
    seed=123,
    batch_size=16,
)


# 1D
trainers_1d = train_models(
    gp_datasets,
    {"AttnCNP": model_1d},
    test_datasets=gp_test_datasets,
    valid_datasets=gp_valid_datasets,
    iterator_train__collate_fn=get_cntxt_trgt_1d,
    iterator_valid__collate_fn=get_cntxt_trgt_1d,
    **KWARGS
)


# 2D
trainers_2d = train_models(
    img_datasets,
    add_y_dim({"AttnCNP": model_2d}, img_datasets),  # y_dim (channels) depend on data
    test_datasets=img_test_datasets,
    train_split=skorch.dataset.CVSplit(0.1),  # use 10% of training for valdiation
    iterator_train__collate_fn=get_cntxt_trgt_2d,
    iterator_valid__collate_fn=get_cntxt_trgt_2d,
    datasets_kwargs=dict(
        zsmms=dict(
            iterator_train__collate_fn=get_cntxt_trgt_2d_extrap,
            iterator_valid__collate_fn=get_cntxt_trgt_2d_extrap,
        )
    ),  # for zsmm use extrapolation
    **KWARGS
)

### Inference

#### GPs Dataset

##### Samples from a single GP

from utils.ntbks_helpers import plot_multi_posterior_samples_1d
from utils.visualize import giffify


def multi_posterior_gp_gif(filename, trainers, datasets, **kwargs):
    giffify(
        f"jupyter/gifs/{filename}.gif",
        gen_single_fig=plot_multi_posterior_samples_1d,
        sweep_parameter="n_cntxt",
        # sweep of context points for GIF
        sweep_values=[1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 20, 25, 30, 40, 50, 75, 100],
        seed=123,  # fix for GIF
        trainers=trainers,
        datasets=datasets,
        is_plot_real=False,  # don't plot sampled function
        plot_config_kwargs=dict(
            set_kwargs=dict(ylim=[-3, 3]), rc={"legend.loc": "upper right"}
        ),  # fix for GIF
        **kwargs,
    )


def filter_single_gp(d):
    return {k: v for k, v in d.items() if ("All" not in k) and ("Vary" not in k)}


multi_posterior_gp_gif(
    "AttnCNP_single_gp",
    trainers=filter_single_gp(trainers_1d),
    datasets=filter_single_gp(gp_datasets),  # will resample from it => not on train
)

```{figure} gifs/AttnCNP_single_gp.gif
---
width: 500px
name: AttnCNP_single_gp
---
[...] understand is how well NPFs can model a ground truth GP [...].
```

From {numref}`AttnCNP_single_gp` we see that not too bad [...], 
But the results are much less nice in the extrapolation regime, as neural networks do not extrapolate well.

multi_posterior_gp_gif(
    "AttnCNP_single_gp_extrap",
    trainers=filter_single_gp(trainers_1d),
    datasets=filter_single_gp(gp_datasets),
    extrap_distance=4,  # add 4 on the right for extrapolation
)

```{figure} gifs/AttnCNP_single_gp_extrap.gif
---
width: 500px
name: AttnCNP_single_gp_extrap
---
[...] understand is how well NPFs can model a ground truth GP [...].
```

From {numref}`AttnCNP_single_gp_extrap` we see that it can clearly not extrapolate.

##### Samples from GPs with varying kernel hyperparameters

def filter_hyp_gp(d):
    return {k: v for k, v in d.items() if ("Vary" in k)}


multi_posterior_gp_gif(
    "AttnCNP_vary_gp",
    trainers=filter_hyp_gp(trainers_1d),
    datasets=filter_hyp_gp(gp_datasets),
)

```{figure} gifs/AttnCNP_vary_gp.gif
---
width: 500px
name: AttnCNP_vary_gp
---
[...] understand is how well NPFs can model a ground truth GP [...].
```

From {numref}`AttnCNP_vary_gp` we see that [...]

##### Samples from GPs with varying Kernels


# data with varying kernels simply merged single kernels
single_gp_datasets = filter_single_gp(gp_datasets)

# use same trainer for all, but have to change their name to be the same as datasets
base_trainer_name = "All_Kernels/AttnCNP/run_0"
trainer = trainers_1d[base_trainer_name]
replicated_trainers = {}
for name in single_gp_datasets.keys():
    replicated_trainers[base_trainer_name.replace("All_Kernels", name)] = trainer

multi_posterior_gp_gif(
    "AttnCNP_kernel_gp", trainers=replicated_trainers, datasets=single_gp_datasets
)

```{figure} gifs/AttnCNP_kernel_gp.gif
---
width: 500px
name: AttnCNP_kernel_gp
---
[...] understand is how well NPFs can model a ground truth GP [...].
```

From {numref}`AttnCNP_kernel_gp` we see that [...]

#### Image Dataset

from utils.ntbks_helpers import plot_multi_posterior_samples_imgs
from utils.visualize import giffify


def multi_posterior_imgs_gif(filename, trainers, datasets, **kwargs):
    giffify(
        f"jupyter/gifs/{filename}.gif",
        gen_single_fig=plot_multi_posterior_samples_imgs,
        sweep_parameter="n_cntxt",
        # sweep of context points for GIF
        sweep_values=[
            0,
            0.001,
            0.003,
            0.005,
            0.007,
            0.01,
            0.02,
            0.03,
            0.05,
            0.07,
            0.1,
            0.15,
            0.2,
            0.3,
            0.5,
            0.7,
            0.99,
            "hhalf",
            "vhalf",
        ],
        seed=123,  # fix for GIF
        trainers=trainers,
        datasets=datasets,
        n_plots=3,  # number of samples plots for each data
        **kwargs,
    )


multi_posterior_imgs_gif(
    "AttnCNP_img", trainers=trainers_2d, datasets=img_test_datasets,
)

```{figure} gifs/AttnCNP_img.gif
---
width: 500px
name: AttnCNP_img
---
[...] dataset img[...].
```

From {numref}`AttnCNP_img` we see that [...]

from utils.ntbks_helpers import PRETTY_RENAMER
from utils.visualize import plot_qualitative_with_kde

n_trainers = len(trainers_2d)
for i, (k, trainer) in enumerate(trainers_2d.items()):
    data_name = k.split("/")[0]
    model_name = k.split("/")[1]
    dataset = img_test_datasets[data_name]

    plot_qualitative_with_kde(
        [PRETTY_RENAMER[model_name], trainer],
        dataset,
        figsize=(9, 7),
        percentiles=[1, 10, 20, 30, 50, 100],
        height_ratios=[1, 5],
        is_smallest_xrange=True,
        h_pad=-4,
        title=PRETTY_RENAMER[data_name],
        test_upscale_factor=get_test_upscale_factor(data_name),
    )

    print(end="\r")
    print(end="\r")

[^selfattn]: To be in line with {numref}`computational_graph_AttnCNPs`, the self attention layers should actually be in the aggregator instead of the encoder. Indeed, we apply the encoder to each context point separately.