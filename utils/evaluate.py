import torch
from skorch.dataset import unpack_data, uses_placeholder_y

from .helpers import set_seed

__all__ = ["eval_loglike"]


def eval_loglike(trainer, dataset, seed=123, **kwargs):
    """Return the log likelihood for each image in order."""
    set_seed(seed)  # make sure same order and indices for cntxt and trgt
    trainer.module_.to(trainer.device)
    old_reduction = trainer.criterion_.reduction
    trainer.criterion_.reduction = None
    y_valid_is_ph = uses_placeholder_y(dataset)
    all_losses = []

    trainer.notify("on_epoch_begin", dataset_valid=dataset)
    for data in trainer.get_iterator(dataset, training=False):
        Xi, yi = unpack_data(data)
        yi_res = yi if not y_valid_is_ph else None
        trainer.notify("on_batch_begin", X=Xi, y=yi_res, training=False)
        step = trainer.validation_step(Xi, yi, **kwargs)
        trainer.notify("on_batch_end", X=Xi, y=yi_res, training=False, **step)
        all_losses.append(-step["loss"])  # use log likelihood instead of NLLL

    trainer.criterion_.reduction = old_reduction
    return torch.cat(all_losses, dim=0).detach().cpu().numpy()
