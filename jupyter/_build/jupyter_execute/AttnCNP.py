# Attentive Conditional Neural Process (AttnCNP)

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import os
import warnings

import matplotlib.pyplot as plt
import torch

os.chdir("..")
# warnings.filterwarnings("ignore")

N_THREADS = 8
IS_FORCE_CPU = False  # Nota Bene : notebooks don't deallocate GPU memory

if IS_FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

torch.set_num_threads(N_THREADS)

from npf.utils.datasplit import CntxtTrgtGetter, GetRandomIndcs, get_all_indcs
from utils.data.dataloader import cntxt_trgt_collate
from utils.ntbks_helpers import get_all_gp_datasets

datasets, test_datasets, valid_datasets = get_all_gp_datasets()

get_cntxt_trgt = CntxtTrgtGetter(
    contexts_getter=GetRandomIndcs(min_n_indcs=0.0, max_n_indcs=0.5),
    targets_getter=get_all_indcs,
)

cntxt_trgt_collate = cntxt_trgt_collate(get_cntxt_trgt)  # make a loader out of it

# CNP Docstring
from npf import AttnCNP

print(AttnCNP.__doc__)

from functools import partial

from npf import AttnCNP
from npf.architectures import MLP, merge_flat_input
from utils.helpers import count_parameters

R_DIM = 128
X_DIM = 1
Y_DIM = 1

model = partial(
    AttnCNP,
    X_DIM,
    Y_DIM,
    r_dim=R_DIM,
    attention="transformer",
    is_self_attn=False,
    XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
        partial(MLP, n_hidden_layers=2, hidden_size=R_DIM), is_sum_merge=True,
    ),
)

n_params = count_parameters(model())
print(f"Number Parameters: {n_params:,d}")

from npf import CNPFLoss
from utils.train import train_models

IS_RETRAIN = False  # whether to load precomputed model or retrain

trainers = train_models(
    datasets,
    {"AttnCNP": model},
    CNPFLoss,  # Standard loss for conditional NPFs
    test_datasets=test_datasets,
    valid_datasets=valid_datasets,
    chckpnt_dirname="results/npfs/1D/",
    is_retrain=IS_RETRAIN,
    iterator_train__collate_fn=cntxt_trgt_collate,
    iterator_valid__collate_fn=cntxt_trgt_collate,
    device=None,  # use GPU if available
    max_epochs=50,
    batch_size=64,
    lr=1e-3,
    decay_lr=10,  # decrease learning rate by 10 during training
    seed=123,
)

### Inference

#### GPs Dataset

##### Samples from a single GP

from utils.ntbks_helpers import plot_multi_posterior_samples_1d
from utils.visualize import giffify


def multi_posterior_gif(filename, trainers, datasets, **kwargs):
    giffify(
        f"jupyter/gifs/{filename}.gif",
        gen_single_fig=plot_multi_posterior_samples_1d,
        sweep_parameter="n_cntxt",
        # sweep of context points for GIF
        sweep_values=[1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100],
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


multi_posterior_gif(
    "AttnCNP_single_gp",
    trainers=filter_single_gp(trainers),
    datasets=filter_single_gp(datasets),  # will resample from it => not on train
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

multi_posterior_gif(
    "AttnCNP_single_gp_extrap",
    trainers=filter_single_gp(trainers),
    datasets=filter_single_gp(datasets),
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


multi_posterior_gif(
    "AttnCNP_vary_gp", trainers=filter_hyp_gp(trainers), datasets=filter_hyp_gp(datasets)
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
single_gp_datasets = filter_single_gp(datasets)

# use same trainer for all, but have to change their name to be the same as datasets
base_trainer_name = "All_Kernels/AttnCNP/run_0"
trainer = trainers[base_trainer_name]
replicated_trainers = {}
for name in single_gp_datasets.keys():
    replicated_trainers[base_trainer_name.replace("All_Kernels", name)] = trainer

multi_posterior_gif(
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

