from npf.utils.helpers import prod


def collapse_z_samples_batch(t):
    """Merge n_z_samples and batch_size in a single dimension."""
    n_z_samples, batch_size, *rest = t.shape
    return t.view(n_z_samples * batch_size, *rest)


def extract_z_samples_batch(t, n_z_samples, batch_size):
    """`reverses` collapse_z_samples_batch."""
    _, *rest = t.shape
    return t.view(n_z_samples, batch_size, *rest)


def replicate_z_samples(t, n_z_samples):
    """Replicates a tensor `n_z_samples` times on a new first dim."""
    return t.unsqueeze(0).expand(n_z_samples, *t.shape)


def pool_and_replicate_middle(t):
    """Mean pools a tensor on all but the first and last dimension (i.e. all the middle dimension)."""
    first, *middle, last = t.shape

    # size = [first, 1, last]
    t = t.view(first, prod(middle), last).mean(1, keepdim=True)

    t = t.view(first, *([1] * len(middle)), last)
    t = t.expand(first, *middle, last)

    # size = [first, *middle, last]
    return t
