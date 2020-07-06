# Conditional Neural Process (CNP)


```{figure} images/computational_graph_CNPs.svg
---
height: 250px
name: computational_graph_CNPs
---
Computational graph for Conditional Neural Processes.
```

CNPs differ from other CNPFs in that they use a mean operator for the aggregator.
The computational graph is thus very simple ({numref}`computational_graph_CNPs`). 

## Properties

CNPs have the following desirable properties compared to other CNPFs:

* &#10003; **$\mathbf{\mathcal{O}(T+C)}$ Inference**. Predicting using the posterior predictive is $\mathcal{O}(T+C)$. I.e. $\mathcal{O}(C)$ to compute $R$ (summarize context) and then $\mathcal{O}(T)$ to predict at each target (conditional independence).

But it suffers from the following issues:

* &#10007; **Underfitting**. The global representation $R$ is a single vector which does not depend on the target features. E.g. predicting very close to context examples will not decrease uncertainty.

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
from utils.data import cntxt_trgt_collate
from utils.ntbks_helpers import get_all_gp_datasets, get_img_datasets

# DATASETS
# merges : get_datasets_single_gp, get_datasets_varying_hyp_gp, get_datasets_varying_kernel_gp
gp_datasets, gp_test_datasets, gp_valid_datasets = get_all_gp_datasets()
# image datasets
img_datasets, img_test_datasets = get_img_datasets(["celeba32", "mnist", "zsmms"])

# CONTEXT TARGET SPLIT
# `CntxtTrgtGetter` : return context and target
# `cntxt_trgt_collate` : make it compatible with pytorch loaders (collate function)
get_cntxt_trgt_1d = cntxt_trgt_collate(
    CntxtTrgtGetter(contexts_getter=GetRandomIndcs(min_n_indcs=0.0, max_n_indcs=0.5))
)
# same as in 1D but with masks (2d) rather than indices
get_cntxt_trgt_2d = cntxt_trgt_collate(
    GridCntxtTrgtGetter(context_masker=RandomMasker(min_nnz=0.0, max_nnz=0.5))
)

Let's now define the models. For both the 1D and 2D case we will be using the following:
* **Encoder** $\mathrm{e}_{\boldsymbol{\theta}}$ : a 1-hidden layer MLP that encodes the features ($\{x^{(i)}\} \mapsto \{x_{transformed}^{(i)}\}$), followed by a 2 hidden layer MLP that encodes each feature-value pair ($\{x_{transformed}^{(i)}, y^{(i)}\} \mapsto \{R^{(i)}\}$).
* **Aggregator** $\mathrm{Agg}$: mean operator.
* **Decoder** $\mathrm{d}_{\boldsymbol{\theta}}$: a 4 hidden layer MLP that predicts the distribution of the target value given the global representation and target context ($\{R, x^{(t)}\} \mapsto \{\mu^{(t)}, \sigma^{2(t)}\}$).

All hidden representations will be of 128 dimensions besides the encoder which is has width $128*2$ for the 1D case and $128*3$ for the 2D case (to have similar number of parameters than other NPFs)s.

For more details about all the possible parameters, refer to the docstrings of `CNP` and the base class `NeuralProcessFamily`.

# CNP Docstring
from npf import CNP

print(CNP.__doc__)

# NeuralProcessFamily Docstring
from npf import NeuralProcessFamily

print(NeuralProcessFamily.__doc__)

from functools import partial

from npf.architectures import MLP, merge_flat_input
from utils.helpers import count_parameters

R_DIM = 128
KWARGS = dict(
    XEncoder=partial(MLP, n_hidden_layers=1, hidden_size=R_DIM),
    Decoder=merge_flat_input(  # MLP takes single input but we give x and R so merge them
        partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
    ),
    r_dim=R_DIM,
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

n_params_1d = count_parameters(model_1d())
n_params_2d = count_parameters(model_2d(y_dim=3))
print(f"Number Parameters (1D): {n_params_1d:,d}")
print(f"Number Parameters (2D): {n_params_2d:,d}")

### Training

The main function for training is `train_models` which trains a dictionary of models on a dictionary of datasets and returns all the trained models.
See its docstring for possible parameters.

# train_models Docstring
from utils.train import train_models

print(train_models.__doc__)

import skorch
from npf import CNPFLoss
from utils.ntbks_helpers import add_y_dim
from utils.train import train_models

KWARGS = dict(
    is_retrain=False,  # whether to load precomputed model or retrain
    criterion=CNPFLoss,  # Standard loss for conditional NPFs
    chckpnt_dirname="results/npfs/ntbks/",
    device=None,  # use GPU if available
    max_epochs=50,
    batch_size=32,
    lr=1e-3,
    decay_lr=10,  # decrease learning rate by 10 during training
    seed=123,
)


# 1D
trainers_1d = train_models(
    gp_datasets,
    {"CNP": model_1d},
    test_datasets=gp_test_datasets,
    valid_datasets=gp_valid_datasets,
    iterator_train__collate_fn=get_cntxt_trgt_1d,
    iterator_valid__collate_fn=get_cntxt_trgt_1d,
    **KWARGS
)


# 2D
trainers_2d = train_models(
    img_datasets,
    add_y_dim({"CNP": model_2d}, img_datasets),  # y_dim (channels) depend on data
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
    "CNP_single_gp",
    trainers=filter_single_gp(trainers_1d),
    datasets=filter_single_gp(gp_datasets),  # will resample from it => not on train
)

```{figure} gifs/CNP_single_gp.gif
---
width: 500px
name: CNP_single_gp
---
[...] understand is how well NPFs can model a ground truth GP [...].
```

From {numref}`CNP_single_gp` we see that [...]

##### Samples from GPs with varying kernel hyperparameters

def filter_hyp_gp(d):
    return {k: v for k, v in d.items() if ("Vary" in k)}


multi_posterior_gp_gif(
    "CNP_vary_gp",
    trainers=filter_hyp_gp(trainers_1d),
    datasets=filter_hyp_gp(gp_datasets),
)

```{figure} gifs/CNP_vary_gp.gif
---
width: 500px
name: CNP_vary_gp
---
[...] understand is how well NPFs can model a ground truth GP [...].
```

From {numref}`CNP_vary_gp` we see that [...]

##### Samples from GPs with varying Kernels

As we have seen in {doc}`data <Datasets>`, the dataset with varying kernel simply merged all the datasets with a single kernel. 
We will now test each separately. To see how the CNP can recover the ground truth GP even though it was trained with samples from different kernels (including the correct one).

# data with varying kernels simply merged single kernels
single_gp_datasets = filter_single_gp(gp_datasets)

# use same trainer for all, but have to change their name to be the same as datasets
base_trainer_name = "All_Kernels/CNP/run_0"
trainer = trainers_1d[base_trainer_name]
replicated_trainers = {}
for name in single_gp_datasets.keys():
    replicated_trainers[base_trainer_name.replace("All_Kernels", name)] = trainer

multi_posterior_gp_gif(
    "CNP_kernel_gp", trainers=replicated_trainers, datasets=single_gp_datasets
)

```{figure} gifs/CNP_kernel_gp.gif
---
width: 500px
name: CNP_kernel_gp
---
[...] understand is how well NPFs can model a ground truth GP [...].
```

From {numref}`CNP_kernel_gp` we see that [...]

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
    "CNP_img", trainers=trainers_2d, datasets=img_test_datasets,
)

```{figure} gifs/CNP_img.gif
---
width: 500px
name: CNP_img
---
[...] dataset img[...].
```

From {numref}`CNP_img` we see that [...]

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

