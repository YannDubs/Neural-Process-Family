import os
import sys
from functools import partial

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

from neuralproc.architectures import MLP, merge_flat_input
from neuralproc.utils.datasplit import CntxtTrgtGetter, GetRandomIndcs, get_all_indcs
from utils.data import DIR_DATA, GPDataset

sys.path.append("..")


### TUTORIAL 1 ###
# DATA
def get_gp_datasets_varying(
    n_samples=10000,
    n_points=128,
    save_file=f"{os.path.join(DIR_DATA, 'gp_dataset.hdf5')}",
    **kwargs,
):
    """
    Return different 1D functions sampled from GPs with the following kernels:
    "rbf", "periodic", "non-stationary", "matern", "noisy-matern" with varying
    hyperparameters.
    """
    datasets = dict()
    kwargs.update(dict(n_samples=n_samples, n_points=n_points, is_vary_kernel_hyp=True))

    def add_dataset_(name, kernel, save_file=save_file):
        if save_file is not None:
            save_file = (save_file, name)
        datasets[name] = GPDataset(kernel=kernel, save_file=save_file, **kwargs)

    add_dataset_("RBF_Kernel", RBF(length_scale_bounds=(0.02, 0.3)))

    add_dataset_(
        "Periodic_Kernel",
        ExpSineSquared(length_scale_bounds=(0.2, 0.5), periodicity_bounds=(0.5, 2.0)),
    )
    add_dataset_("Matern_Kernel", Matern(length_scale_bounds=(0.03, 0.3), nu=1.5))
    add_dataset_(
        "Noisy_Matern_Kernel",
        (
            WhiteKernel(noise_level_bounds=(0.05, 0.7))
            + Matern(length_scale_bounds=(0.03, 0.3), nu=1.5)
        ),
    )

    datasets_test = {
        k: skorch.dataset.Dataset(
            *dataset.get_samples(save_file=save_file, idx_chunk=-1)
        )
        for k, dataset in datasets.items()
    }

    return datasets, datasets_test


# MODEL
get_cntxt_trgt = CntxtTrgtGetter(
    contexts_getter=GetRandomIndcs(min_n_indcs=0.01, max_n_indcs=0.5),
    targets_getter=get_all_indcs,
    is_add_cntxts_to_trgts=False,
)

R_DIM = 128
CNP_KWARGS = dict(
    XEncoder=partial(MLP, n_hidden_layers=1, hidden_size=R_DIM),
    XYEncoder=merge_flat_input(
        partial(MLP, n_hidden_layers=2, hidden_size=R_DIM), is_sum_merge=True
    ),
    Decoder=merge_flat_input(
        partial(MLP, n_hidden_layers=2, is_force_hid_smaller=True, hidden_size=R_DIM),
        is_sum_merge=True,
    ),
    r_dim=128,
    encoded_path="deterministic",
)
