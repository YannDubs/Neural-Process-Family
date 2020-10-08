import torch

__all__ = ["cntxt_trgt_collate"]


def cntxt_trgt_collate(get_cntxt_trgt, is_duplicate_batch=False, **kwargs):
    """Transformes and collates inputs to neural processes given the whole input.

    Parameters
    ----------
    get_cntxt_trgt : callable
        Function that takes as input the features and tagrets `X`, `y` and return
        the corresponding `X_cntxt, Y_cntxt, X_trgt, Y_trgt`.

    is_duplicate_batch : bool, optional
        Wether to repeat th`e batch to have 2 different context and target sets
        for every function. If so the batch will contain the concatenation of both.
    """

    def mycollate(batch):
        collated = torch.utils.data.dataloader.default_collate(batch)
        X = collated[0]
        y = collated[1]

        if is_duplicate_batch:
            X = torch.cat([X, X], dim=0)
            if y is not None:
                y = torch.cat([y, y], dim=0)
            y = torch.cat([y, y], dim=0)

        X_cntxt, Y_cntxt, X_trgt, Y_trgt = get_cntxt_trgt(X, y, **kwargs)
        inputs = dict(X_cntxt=X_cntxt, Y_cntxt=Y_cntxt, X_trgt=X_trgt, Y_trgt=Y_trgt)
        targets = Y_trgt

        return inputs, targets

    return mycollate
