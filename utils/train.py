import logging
import os
import warnings
from copy import deepcopy
from functools import partial

import numpy as np
import skorch
import torch
from skorch import NeuralNet
from skorch.callbacks import (
    Checkpoint,
    EarlyStopping,
    LoadInitState,
    LRScheduler,
    ProgressBar,
)
from skorch.helper import predefined_split
from skorch.utils import to_tensor
from torch.optim import Adam

from .evaluate import eval_loglike
from .helpers import FixRandomSeed, get_exponential_decay_gamma

__all__ = ["train_models"]

EVAL_FILENAME = "eval.csv"
MOD_SUMM_FILENAME = "model_summary.txt"

logger = logging.getLogger(__name__)


def train_models(
    datasets,
    models,
    criterion,
    test_datasets=dict(),
    valid_datasets=dict(),
    chckpnt_dirname=None,
    is_continue_train=False,
    is_retrain=False,
    runs=1,
    starting_run=0,
    train_split=skorch.dataset.CVSplit(0.1),
    device=None,
    max_epochs=100,
    batch_size=16,
    lr=1e-3,
    optimizer=Adam,
    callbacks=[ProgressBar()],
    patience=None,
    decay_lr=None,
    is_reeval=False,
    seed=None,
    datasets_kwargs=dict(),
    models_kwargs=dict(),
    **kwargs
):
    """
    Train or loads the models.

    Parameters
    ----------
    datasets : dict
        The datasets on which to train the models. 

    models : dict
        The models to train (initialized or not). Each model will be trained on
        all datasets. If the initialzed models are passed, it will continue
        training from there.  Can also give a dictionary of dictionaries, if the
        models to train depend on the dataset.

    criterion : nn.Module
        The uninitialized criterion (loss).

    test_datasets : dict, optional
        The test datasets. If given, the corresponding models will be evaluated
        on those, the log likelihood for each datapoint will be saved in the
        in the checkpoint directory as `eval.csv`.

    valid_datasets : dict, optional
        The validation datasets. 

    chckpnt_dirname : str, optional
        Directory where checkpoints will be saved. The best (if validation or train_split given)
        or last model will be saved.

    is_continue_train : bool, optional
        Whether to continue training from the last checkpoint of the previous run. 

    is_retrain : bool, optional
        Whether to retrain the model. If not, `chckpnt_dirname` should be given
        to load the pretrained model.

    runs : int, optional
        How many times to run the model. Each run will be saved in
        `chckpnt_dirname/run_{}`. If a seed is give, it will be incremented at
        each run.

    starting_run : int, optional
        Starting run. This is useful if a couple of runs have already been trained,
        and you want to continue from there.

    train_split : callable, optional
        If None, there is no train/validation split. Else, train_split
        should be a function or callable that is called with X and y
        data and should return the tuple ``dataset_train, dataset_valid``.
        The validation data may be None. Use `skorch.dataset.CVSplit` to randomly
        split the data into train and validation. Only used for datasets that are not in 
        `valid_datasets.`.

    device : str, optional
        The compute device to be used (input to torch.device). If `None` uses
        "cuda" if available else "cpu".

    max_epochs : int, optional
        Maximum number of epochs.

    batch_size : int, optional
        Training batch size.

    lr : float, optional
        Learning rate.

    optimizer : torch.optim.Optimizer, optional
        Optimizer.

    callbacks : list, optional
        Callbacks to use.

    patience : int, optional
        Patience for early stopping. If not `None` has to be given a validation
        set.

    decay_lr : float, optional
        Factor by which to decay the learning rate during training. For example if 100 then it
        will decrease the learning rate with exponential decrease such that at the end of training 
        the learning rate decreased by a factot 100.

    is_reeval : bool, optional
        Whether to reevaluate the model even if already evaluated and `is_retrain` is False.
    
    seed : int, optional
        Pseudo random seed to force deterministic results (on CUDA might still
        differ a little).

    datasets_kwargs : dict, optional
        Dictionary of datasets specific kwargs.

    models_kwargs : dict, optional
        Dictionary of model specific kwargs.

    kwargs :
        Additional arguments to `NeuralNet`.

    """
    trainers = dict()
    callbacks_dflt = callbacks
    init_chckpnt_dirname = chckpnt_dirname
    NeuralNet.get_loss = get_loss  # ensures that .is_training works

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if "iterator_train" not in kwargs and "iterator_train__shuffle" not in kwargs:
        # default to shuffle when using default iterator
        kwargs["iterator_train__shuffle"] = True

    if "iterator_valid" not in kwargs and "iterator_valid__batch_size" not in kwargs:
        # default to 2 times batch size to be sure to be runnable (no backprop)
        kwargs["iterator_valid__batch_size"] = batch_size * 2

    for data_name, data_train in datasets.items():

        if isinstance(list(models.values())[0], dict):
            # if dict of dict then depends on data
            current_models = models[data_name]
        else:
            current_models = models

        data_test = test_datasets.get(data_name, None)
        data_valid = valid_datasets.get(data_name, None)
        curr_train_split = (
            train_split if data_valid is None else predefined_split(data_valid)
        )

        for model_name, model in current_models.items():

            for run in range(starting_run, starting_run + runs):

                callbacks = deepcopy(callbacks_dflt)

                suffix = data_name + "/" + model_name + "/run_{}".format(run)

                print(
                    "\n--- {} {} ---\n".format(
                        "Training" if is_retrain else "Loading", suffix
                    ),
                    flush=True,
                )

                if chckpnt_dirname is not None:
                    chckpnt_dirname = init_chckpnt_dirname + suffix
                    test_eval_file = os.path.join(chckpnt_dirname, EVAL_FILENAME)
                    if curr_train_split is None:
                        # checkpoints only last
                        chckpt = Checkpoint(dirname=chckpnt_dirname, monitor=None)
                    else:
                        chckpt = Checkpoint(
                            dirname=chckpnt_dirname, monitor="valid_loss_best"
                        )
                    callbacks.extend([chckpt])

                if is_continue_train:
                    assert chckpnt_dirname is not None
                    # start from chckpt until LastCheckpoint is implemented
                    load_state = LoadInitState(chckpt)
                    callbacks.append(load_state)

                if patience is not None:
                    callbacks.append(EarlyStopping(patience=patience))

                if seed is not None:
                    # make sure that the seed changes across runs
                    callbacks.append(FixRandomSeed(seed + run))

                if decay_lr is not None:
                    gamma = get_exponential_decay_gamma(decay_lr, max_epochs)
                    callbacks.append(
                        LRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=gamma)
                    )

                kwargs.update(
                    dict(
                        train_split=curr_train_split,
                        warm_start=True,  # continue training
                        callbacks=callbacks,
                        device=device,
                        optimizer=optimizer,
                        lr=lr,
                        max_epochs=max_epochs,
                        batch_size=batch_size,
                    )
                )

                current_kwargs = kwargs.copy()
                current_kwargs.update(datasets_kwargs.get(data_name, dict()))
                current_kwargs.update(models_kwargs.get(model_name, dict()))

                trainer = NeuralNet(model, criterion, **current_kwargs)

                if is_retrain:
                    _ = trainer.fit(data_train)
                    with open(
                        os.path.join(chckpnt_dirname, MOD_SUMM_FILENAME), "w"
                    ) as f:
                        f.write(str(trainer.module_))

                # load in all case => even when training loads the best checkpoint
                trainer.initialize()
                trainer.load_params(checkpoint=chckpt)

                # return the training rather than testing history as dflt
                history = deepcopy(trainer.history)
                test_loglike = _eval_save_load(
                    trainer,
                    data_test,
                    test_eval_file,
                    is_force_rerun=is_retrain or is_reeval,
                )
                trainer.test_history = deepcopy(trainer.history)
                trainer.history = history

                valid_loss, best_epoch = _best_loss(trainer, mode="valid")
                train_loss, _ = _best_loss(trainer, mode="train")

                print(
                    suffix,
                    "| best epoch:",
                    best_epoch,
                    "| train loss:",
                    round_decimals(train_loss, n=4),
                    "| valid loss:",
                    round_decimals(valid_loss, n=4),
                    "| test log likelihood:",
                    round_decimals(test_loglike, n=4),
                    flush=True,
                )

                trainer.module_.cpu()  # make sure on cpu
                torch.cuda.empty_cache()  # empty cache for next run

                trainers[suffix] = trainer

    return trainers


def round_decimals(x, n=4):
    if x is not None:
        pattern = "{:." + str(n) + "f}"
        x = float(pattern.format(x))
    return x


def _eval_save_load(model, data_test, test_eval_file, is_force_rerun=False):
    test_loglike = None

    if data_test is not None:

        if test_eval_file is not None and os.path.exists(test_eval_file):
            test_loglike = np.loadtxt(test_eval_file, delimiter=",")

        if is_force_rerun or test_loglike is None:
            test_loglike = eval_loglike(model, data_test)

        if test_eval_file is not None:
            np.savetxt(test_eval_file, test_loglike, delimiter=",")

        return test_loglike.mean(axis=0)


def _best_loss(model, mode="valid"):
    try:
        for epoch, history in enumerate(model.history[::-1]):
            if history["{}_loss_best".format(mode)]:
                best_epoch = len(model.history) - epoch
                return history["{}_loss".format(mode)], best_epoch
    except:
        return None, None


def get_loss(self, y_pred, y_true, X=None, training=False):
    """Return the loss for this batch."""
    y_true = to_tensor(y_true, device=self.device)

    if isinstance(self.criterion_, torch.nn.Module):
        self.criterion_.train(training)

    return self.criterion_(y_pred, y_true)
