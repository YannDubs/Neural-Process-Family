import contextlib
import glob
import logging
import os
import random
from multiprocessing import Pool, cpu_count

import IPython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from skorch.callbacks import Callback


def mean(l):
    """Average over array like."""
    return sum(l) / len(l)


def load_all_results(folder):
    """Load all the results (by data, model, run) from a directory."""
    pattern = "*/*/run_*/eval.csv"
    df = pd.DataFrame(
        [
            f.split("/")[-4:-1] + [pd.read_csv(f, header=None).mean()[0]]
            for f in glob.glob(os.path.join(folder, pattern))
        ]
    )
    df.columns = ["Data", "Model", "Runs", "LogLike"]
    return df


def get_exponential_decay_gamma(scheduling_factor, max_epochs):
    """Return the exponential learning rate factor gamma.

    Parameters
    ----------
    scheduling_factor :
        By how much to reduce learning rate during training.

    max_epochs : int
        Maximum number of epochs.
    """
    return (1 / scheduling_factor) ** (1 / max_epochs)


def set_seed(seed):
    """Set the random seed."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


class FixRandomSeed(Callback):
    """
    Callback to have a deterministic behavior.
    Credits: https://github.com/skorch-dev/skorch/issues/280
    """

    def __init__(self, seed=123, is_cudnn_deterministic=False, verbose=0):
        self.seed = seed
        self.is_cudnn_deterministic = is_cudnn_deterministic
        self.verbose = verbose

    def initialize(self):
        if self.seed is not None:
            if self.verbose > 0:
                print("setting random seed to: ", self.seed, flush=True)
            set_seed(self.seed)
        torch.backends.cudnn.deterministic = self.is_cudnn_deterministic


def parallelize(data, func, axis_split=0, n_chunks=None, cores=cpu_count()):
    """Run a function in parallel on a numpy array `data`, by splitting it in chunks."""
    if n_chunks is None:
        n_chunks = cores * 2
    data_split = np.array_split(data, n_chunks, axis=axis_split)
    pool = Pool(cores)
    outs = pool.map(func, data_split)
    if isinstance(outs[0], tuple):
        # multioutput
        outs = tuple_cont_to_cont_tuple(outs)
        data = tuple(np.concatenate(out, axis=axis_split) for out in outs)
    else:
        data = np.concatenate(outs, axis=axis_split)

    pool.close()
    pool.join()
    return data


class DisableLogger:
    def __enter__(self):
        logging.disable(50)

    def __exit__(self, a, b, c):
        logging.disable(logging.NOTSET)


@contextlib.contextmanager
def silent():
    """Context manager to remove all outputs and warnings."""
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(
        f
    ), DisableLogger(), IPython.utils.io.capture_output():
        yield


def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum([p.numel() for p in model.parameters()])


def get_only_first_item(to_index):
    """Helper function to make a class `to_index` return `to_index[i][0]` when indexed."""

    class FirstIndex:
        def __init__(self, to_index):
            self.to_index = to_index

        def __getitem__(self, i):
            return self.to_index[i][0]

        def __len__(self):
            return len(self.to_index)

    return FirstIndex(to_index)


def make_Xy_input(dataset, y=None):
    """
    Transform a dataset X to a variable that can be directly used like so:
    `NeuralNetEstimator.fit(*make_Xy_input(dataset))` when both `X` and `y`
    should be inputs to `forward`. Can also give a X and y.
    """
    if isinstance(dataset, dict):
        y = dataset["y"]
        X = dataset["X"]
    elif isinstance(dataset, torch.utils.data.Dataset):
        if y is None:
            try:
                y = dataset.targets
            except AttributeError:
                y = dataset.y  # skorch datasets
        X = get_only_first_item(dataset)
    else:
        # array-like or tensor
        X = dataset

    return ({"X": X, "y": y}, y)


def tuple_cont_to_cont_tuple(tuples):
    """Converts a tuple of containers (list, tuple, dict) to a container of tuples."""
    if isinstance(tuples[0], dict):
        # assumes keys are correct
        return {k: tuple(dic[k] for dic in tuples) for k in tuples[0].keys()}
    elif isinstance(tuples[0], list):
        return list(zip(*tuples))
    elif isinstance(tuples[0], tuple):
        return tuple(zip(*tuples))
    else:
        raise ValueError("Unkown conatiner type: {}.".format(type(tuples[0])))
