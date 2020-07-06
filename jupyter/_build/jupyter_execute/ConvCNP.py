# Convolutional Conditional Neural Process (ConvCNP)

```{figure} images/computational_graph_ConvCNPs.svg
---
height: 250px
name: computational_graph_ConvCNPs
---
Computational graph for Convolutional Conditional Neural Processes.
```

AttnCNPs differ from other CNPFs in that they model stationary stochastic processes : translation equivariant, functional representation, convolutions, set convolutions, induced points, [...]


## Properties

CNPs have the following desirable properties compared to other CNPFs:

* &#10003; **Translation equivariant**. Inductive bias, sample efficient [...]
* &#10003; **Can Extrapolate**. Due to translation equiariance [...]

But it suffers from the following issues:

* &#10007; **$\mathbf{\mathcal{O}(U(T+C))}$ Inference**. Indeed, the first set convolution compares every context point with every pseudo-inputs $\mathcal{O}(U+C))$. The last set convolution compares every induced points with every target points $\mathcal{O}(U+T))$.
The ConvCNP is thus more computationally efficient than {doc}`AttnCNPs <ANP>` if $U < C$.



%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import os
import warnings
import logging

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
from utils.data import cntxt_trgt_collate
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
    GridCntxtTrgtGetter(context_masker=RandomMasker(min_nnz=0.0, max_nnz=0.5)),
    is_return_masks=True,  # will be using grid conv CNP => can work directly with mask
)

Let's now define the models. For both the 1D and 2D case we will be using the following:
* **Encoder** $\mathrm{e}_{\boldsymbol{\theta}}$ : [...]
* **Aggregator** $\mathrm{Agg}$: [...]
* **Decoder** $\mathrm{d}_{\boldsymbol{\theta}}$:[...]

All hidden representations will be of 128 dimensions.

For more details about all the possible parameters, refer to the docstrings of `ConvCNP` and `GridConvCNP` and the base class `NeuralProcessFamily`.

# ConvCNP Docstring
from npf import ConvCNP

print(ConvCNP.__doc__)

# GridConvCNP Docstring
from npf import GridConvCNP

print(GridConvCNP.__doc__)

from functools import partial

from npf.architectures import CNN, ResConvBlock
from utils.helpers import count_parameters

R_DIM = 128
CNN_KWARGS = dict(
    ConvBlock=ResConvBlock,
    n_blocks=5,
    is_chan_last=True,  # all computations are done with channel last in our code
    kernel_size=9,
    n_conv_layers=2,
)


# 1D case
model_1d = partial(
    ConvCNP,
    x_dim=1,
    y_dim=1,
    CNN=partial(
        CNN, Conv=torch.nn.Conv1d, Normalization=torch.nn.BatchNorm1d, **CNN_KWARGS
    ),
    n_induced=256,  # size of discretization
    r_dim=R_DIM,
)


# image (2D) case
model_2d = partial(
    GridConvCNP,
    x_dim=1,  # for gridded conv it's the mask shape
    CNN=partial(
        CNN, Conv=torch.nn.Conv2d, Normalization=torch.nn.BatchNorm2d, **CNN_KWARGS
    ),
    r_dim=R_DIM,
)

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
    device="cuda:1",
    max_epochs=50,
    lr=1e-3,
    decay_lr=10,
    seed=123,
    batch_size=32,
)


# 1D
trainers_1d = train_models(
    gp_datasets,
    {"ConvCNP": model_1d},
    test_datasets=gp_test_datasets,
    valid_datasets=gp_valid_datasets,
    iterator_train__collate_fn=get_cntxt_trgt_1d,
    iterator_valid__collate_fn=get_cntxt_trgt_1d,
    **KWARGS
)



# 2D
trainers_2d = train_models(
    img_datasets,
    add_y_dim({"ConvCNP": model_2d}, img_datasets),  # y_dim (channels) depend on data
    test_datasets=img_test_datasets,
    train_split=skorch.dataset.CVSplit(0.1),  # use 10% of training for valdiation
    iterator_train__collate_fn=get_cntxt_trgt_2d,
    iterator_valid__collate_fn=get_cntxt_trgt_2d,
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
    "ConvCNP_single_gp_extrap",
    trainers=filter_single_gp(trainers_1d),
    datasets=filter_single_gp(gp_datasets),  # will resample from it => not on train
    extrap_distance=4,  # add 4 on the right for extrapolation
)

```{figure} gifs/ConvCNP_single_gp_extrap.gif
---
width: 500px
name: ConvCNP_single_gp_extrap
---
[...] understand is how well NPFs can model a ground truth GP [...].
```

From {numref}`ConvCNP_single_gp` we see that not too bad [...], 
But the results are much less nice in the extrapolation regime, as neural networks do not extrapolate well.

##### Samples from GPs with varying kernel hyperparameters

def filter_hyp_gp(d):
    return {k: v for k, v in d.items() if ("Vary" in k)}


multi_posterior_gp_gif(
    "ConvCNP_vary_gp",
    trainers=filter_hyp_gp(trainers_1d),
    datasets=filter_hyp_gp(gp_datasets),
)

```{figure} gifs/ConvCNP_vary_gp.gif
---
width: 500px
name: ConvCNP_vary_gp
---
[...] understand is how well NPFs can model a ground truth GP [...].
```

From {numref}`ConvCNP_vary_gp` we see that [...]

##### Samples from GPs with varying Kernels

# data with varying kernels simply merged single kernels
single_gp_datasets = filter_single_gp(gp_datasets)

# use same trainer for all, but have to change their name to be the same as datasets
base_trainer_name = "All_Kernels/ConvCNP/run_0"
trainer = trainers_1d[base_trainer_name]
replicated_trainers = {}
for name in single_gp_datasets.keys():
    replicated_trainers[base_trainer_name.replace("All_Kernels", name)] = trainer

multi_posterior_gp_gif(
    "ConvCNP_kernel_gp", trainers=replicated_trainers, datasets=single_gp_datasets
)

```{figure} gifs/ConvCNP_kernel_gp.gif
---
width: 500px
name: ConvCNP_kernel_gp
---
[...] understand is how well NPFs can model a ground truth GP [...].
```

From {numref}`ConvCNP_kernel_gp` we see that [...]

#### Image Dataset

from utils.ntbks_helpers import plot_multi_posterior_samples_imgs, plot_posterior_img
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
    "ConvCNP_img", trainers=trainers_2d, datasets=img_test_datasets,
)

```{figure} gifs/ConvCNP_img.gif
---
width: 500px
name: ConvCNP_img
---
[...] dataset img[...].
```

From {numref}`ConvCNP_img` we see that [...]

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
    )

    print(end="\r")
    print(end="\r")

