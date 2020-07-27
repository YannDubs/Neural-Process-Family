import functools
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import betabinom

from .helpers import (
    channels_to_2nd_dim,
    channels_to_last_dim,
    indep_shuffle_,
    prod,
    ratio_to_int,
)

__all__ = [
    "get_all_indcs",
    "GetRangeIndcs",
    "GetRandomIndcs",
    "CntxtTrgtGetter",
    "RandomMasker",
    "half_masker",
    "no_masker",
    "GridCntxtTrgtGetter",
]


### INDICES SELECTORS ###
def get_all_indcs(batch_size, n_possible_points):
    """
    Return all possible indices.
    """
    return torch.arange(n_possible_points).expand(batch_size, n_possible_points)


class GetRangeIndcs:
    """Get all indices in a certain range."""

    def __init__(self, arange):
        self.arange = arange

    def __call__(self, batch_size, n_possible_points):
        indcs = torch.arange(*self.arange)
        return indcs.expand(batch_size, len(indcs))


class GetIndcsMerger:
    """Meta indexer that merges indices from multiple indexers."""

    def __init__(self, indexers):
        self.indexers = indexers

    def __call__(self, batch_size, n_possible_points):
        indcs = [indexer(batch_size, n_possible_points) for indexer in self.indexers]
        indcs = torch.cat(indcs, dim=1)
        return indcs


class GetRandomIndcs:
    """
    Return random subset of indices.

    Parameters
    ----------
    a : float or int, optional
        Minimum number of indices. If smaller than 1, represents a percentage of
        points.

    b : float or int, optional
        Maximum number of indices. If smaller than 1, represents a percentage of
        points.

    is_batch_share : bool, optional
        Whether to use use the same indices for all elements in the batch.

    range_indcs : tuple, optional
        Range tuple (max, min) for the indices.
        
    is_beta_binomial : bool, optional
        Whether to use beta binomial distribution instead of uniform. In this case a and b become
        respectively alpha and beta in beta binomial distributions. For example to have a an 
        exponentially decaying pdf with a median around 5% use alpha 1 and beta 14.

    proba_uniform : float, optional
        Probability [0,1] of randomly sampling any number of indices regardless of a and b. Useful to 
        ensure that the support is all possible indices.
    """

    def __init__(
        self,
        a=0.1,
        b=0.5,
        is_batch_share=False,
        range_indcs=None,
        is_ensure_one=False,
        is_beta_binomial=False,
        proba_uniform=0,
    ):
        self.a = a
        self.b = b
        self.is_batch_share = is_batch_share
        self.range_indcs = range_indcs
        self.is_ensure_one = is_ensure_one
        self.is_beta_binomial = is_beta_binomial
        self.proba_uniform = proba_uniform

    def __call__(self, batch_size, n_possible_points):
        if self.range_indcs is not None:
            n_possible_points = self.range_indcs[1] - self.range_indcs[0]

        if np.random.uniform(size=1) < self.proba_uniform:
            # whether to sample from a uniform distribution instead of using a and b
            n_indcs = random.randint(0, n_possible_points)

        else:
            if self.is_beta_binomial:
                rv = betabinom(n_possible_points, self.a, self.b)
                n_indcs = rv.rvs()

            else:
                a = ratio_to_int(self.a, n_possible_points)
                b = ratio_to_int(self.b, n_possible_points)
                n_indcs = random.randint(a, b)

        if self.is_ensure_one and n_indcs < 1:
            n_indcs = 1

        if self.is_batch_share:
            indcs = torch.randperm(n_possible_points)[:n_indcs]
            indcs = indcs.unsqueeze(0).expand(batch_size, n_indcs)
        else:
            indcs = (
                np.arange(n_possible_points)
                .reshape(1, n_possible_points)
                .repeat(batch_size, axis=0)
            )
            indep_shuffle_(indcs, -1)
            indcs = torch.from_numpy(indcs[:, :n_indcs])

        if self.range_indcs is not None:
            # adding is teh same as shifting
            indcs += self.range_indcs[0]

        return indcs


class CntxtTrgtGetter:
    """
    Split a dataset into context and target points based on indices.

    Parameters
    ----------
    contexts_getter : callable, optional
        Get the context indices if not given directly (useful for training).

    targets_getter : callable, optional
        Get the context indices if not given directly (useful for training).

    is_add_cntxts_to_trgts : bool, optional
        Whether to add the context points to the targets.
    """

    def __init__(
        self,
        contexts_getter=GetRandomIndcs(),
        targets_getter=get_all_indcs,
        is_add_cntxts_to_trgts=False,
    ):
        self.contexts_getter = contexts_getter
        self.targets_getter = targets_getter
        self.is_add_cntxts_to_trgts = is_add_cntxts_to_trgts

    def __call__(
        self, X, y=None, context_indcs=None, target_indcs=None, is_return_indcs=False
    ):
        """
        Parameters
        ----------
        X: torch.Tensor, size = [batch_size, num_points, x_dim]
            Position features. Values should always be in [-1, 1].

        Y: torch.Tensor, size = [batch_size, num_points, y_dim]
            Targets.

        context_indcs : np.array, size=[batch_size, n_indcs]
            Indices of the context points. If `None` generates it using
            `contexts_getter(batch_size, num_points)`.

        target_indcs : np.array, size=[batch_size, n_indcs]
            Indices of the target points. If `None` generates it using
            `contexts_getter(batch_size, num_points)`.

        is_return_indcs : bool, optional
            Whether to return X and the selected context and taregt indices, rather
            than the selected `X_cntxt, Y_cntxt, X_trgt, Y_trgt`.
        """
        batch_size, num_points = self.getter_inputs(X)

        if context_indcs is None:
            context_indcs = self.contexts_getter(batch_size, num_points)
        if target_indcs is None:
            target_indcs = self.targets_getter(batch_size, num_points)

        if self.is_add_cntxts_to_trgts:
            target_indcs = self.add_cntxts_to_trgts(
                num_points, target_indcs, context_indcs
            )

        # only used if X for context and target should be different (besides selecting indices!)
        X_pre_cntxt = self.preprocess_context(X)

        if is_return_indcs:
            # instead of features return indices / masks, and `Y_cntxt` is replaced
            # with all values Y
            return (
                context_indcs,
                X_pre_cntxt,
                target_indcs,
                X,
            )

        X_cntxt, Y_cntxt = self.select(X_pre_cntxt, y, context_indcs)
        X_trgt, Y_trgt = self.select(X, y, target_indcs)
        return X_cntxt, Y_cntxt, X_trgt, Y_trgt

    def preprocess_context(self, X):
        """Preprocess the data for the context set."""
        return X

    def add_cntxts_to_trgts(self, num_points, target_indcs, context_indcs):
        """
        Add context points to targets. This might results in duplicate indices in
        the targets.
        """
        target_indcs = torch.cat([target_indcs, context_indcs], dim=-1)
        # to reduce the probability of duplicating indices remove context indices
        # that made target indices larger than n_possible_points
        return target_indcs[:, :num_points]

    def getter_inputs(self, X):
        """Make the input for the getters."""
        batch_size, num_points, x_dim = X.shape
        return batch_size, num_points

    def select(self, X, y, indcs):
        """Select the correct values from X."""
        batch_size, num_points, x_dim = X.shape
        y_dim = y.size(-1)
        indcs_x = indcs.to(X.device).unsqueeze(-1).expand(batch_size, -1, x_dim)
        indcs_y = indcs.to(X.device).unsqueeze(-1).expand(batch_size, -1, y_dim)
        return (
            torch.gather(X, 1, indcs_x).contiguous(),
            torch.gather(y, 1, indcs_y).contiguous(),
        )


### GRID AND MASKING ###
class RandomMasker(GetRandomIndcs):
    """
    Return random subset mask.
    """

    def __call__(self, batch_size, mask_shape, **kwargs):
        n_possible_points = prod(mask_shape)
        nnz_indcs = super().__call__(batch_size, n_possible_points, **kwargs)

        if self.is_batch_share:
            # share memory
            mask = torch.zeros(n_possible_points).bool()
            mask = mask.unsqueeze(0).expand(batch_size, n_possible_points)
        else:
            mask = torch.zeros((batch_size, n_possible_points)).bool()

        mask.scatter_(1, nnz_indcs, 1)
        mask = mask.view(batch_size, *mask_shape, 1).contiguous()

        return mask


class ResolutionMasker:
    """
    Return mask that corresponds to decreasing the resolution.

    Parameters
    ----------
    factor : int, optional
        Factor by which to decrease the resolution.
    """

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, batch_size, mask_shape):
        mask = torch.zeros(mask_shape).bool()
        mask[self.factor // 2 :: self.factor, self.factor // 2 :: self.factor] = True
        # share memory
        return mask.unsqueeze(-1).expand(batch_size, *mask_shape, 1)


def and_masks(*masks):
    """Composes tuple of masks by an and operation."""
    mask = functools.reduce(lambda a, b: a & b, masks)
    return mask


def or_masks(*masks):
    """Composes tuple of masks by an or operation."""
    mask = functools.reduce(lambda a, b: a | b, masks)
    return mask


def not_masks(mask, not_mask):
    """Keep all elements in first mask that are nto in second."""
    mask = and_masks(mask, ~not_mask)
    return mask


def half_masker(batch_size, mask_shape, dim=0):
    """Return a mask which masks the top half features of `dim`."""
    mask = torch.zeros(mask_shape).bool()
    slcs = [slice(None)] * (len(mask_shape))
    slcs[dim] = slice(0, mask_shape[dim] // 2)
    mask[slcs] = 1
    # share memory
    return mask.unsqueeze(-1).expand(batch_size, *mask_shape, 1)


def no_masker(batch_size, mask_shape):
    """Return a mask of all 1."""
    mask = torch.ones(1).bool()
    # share memory
    return mask.expand(batch_size, *mask_shape, 1)


class GridCntxtTrgtGetter(CntxtTrgtGetter):
    """
    Split grids of values (e.g. images) into context and target points.

    Parameters
    ----------
    context_masker : callable, optional
        Get the context masks if not given directly (useful for training).

    target_masker : callable, optional
        Get the context masks if not given directly (useful for training).

    upscale_factor : float, optional 
        Factor by which to upscale the image. If 1 then all the images are seen as the same 
        ``size'' regardless of the number of pixels (because all set to be in -1 1)
    
    kwargs:
        Additional arguments to `CntxtTrgtGetter`.
    """

    #! target mask not woeking for now because assuming all the grid is target

    def __init__(
        self,
        context_masker=RandomMasker(),
        target_masker=no_masker,
        upscale_factor=1,
        **kwargs
    ):
        self.upscale_factor = upscale_factor
        super().__init__(
            contexts_getter=context_masker, targets_getter=target_masker, **kwargs
        )

    def __call__(
        self,
        X,
        y=None,
        context_mask=None,
        target_mask=None,
        is_return_masks=False,
        **kwargs
    ):
        """
        Parameters
        ----------
        X: torch.Tensor, size=[batch_size, y_dim, *grid_shape]
            Grid input. Batch where the first dimension represents the number
            of outputs, the rest are the grid dimensions. E.g. (3, 32, 32)
            for images would mean output is 3 dimensional (channels) while
            features are on a 2d grid.

        y: None
            Placeholder

        context_mask : torch.BoolTensor, size=[batch_size, *grid_shape]
            Binary mask indicating the context. Number of non zero should
            be same for all batch. If `None` generates it using
            `context_masker(batch_size, mask_shape)`.

        target_mask : torch.BoolTensor, size=[batch_size, *grid_shape]
            Binary mask indicating the targets. Number of non zero should
            be same for all batch. If `None` generates it using
            `target_masker(batch_size, mask_shape)`.

        is_return_masks : bool, optional
            Whether to return X and the context and target masks, rather
            than the selected `X_cntxt, Y_cntxt, X_trgt, Y_trgt`.
        """
        # parent function assumes y_dim is last rank
        return super().__call__(
            channels_to_last_dim(X),
            context_indcs=context_mask,
            target_indcs=target_mask,
            is_return_indcs=is_return_masks,
            **kwargs
        )

    def add_cntxts_to_trgts(self, grid_shape, target_mask, context_mask):
        """Add context points to targets: has been shown emperically better."""
        return or_masks(target_mask, context_mask)

    def getter_inputs(self, X):
        """Make the input for the getters."""
        batch_size, *grid_shape, y_dim = X.shape
        return batch_size, grid_shape

    def select(self, X, y, mask, extrapolation=1):
        """
        Applies a batch of mask to a grid of size=[batch_size, *grid_shape, y_dim],
        and return a the masked X and Y values. `y` is a placeholder.
        """

        batch_size, *grid_shape, y_dim = X.shape
        n_grid_dim = len(grid_shape)

        # make sure on correct device
        device = X.device
        mask = mask.to(device)

        # batch_size, x_dim
        nonzero_idcs = mask.nonzero()
        # assume same amount of nonzero across batch
        n_cntxt = mask[0].nonzero().size(0)

        # first dim is batch idx and last is y_dim => take in between to get grid
        X_masked = nonzero_idcs[:, 1:-1].view(batch_size, n_cntxt, n_grid_dim).float()
        # normalize grid idcs to [-1,1]
        for i, size in enumerate(grid_shape):
            X_masked[:, :, i] *= 2 / (size - 1)  # in [0,2]
            X_masked[:, :, i] -= 1  # in [-1,1]
        X_masked *= self.upscale_factor

        mask = mask.expand(batch_size, *grid_shape, y_dim)
        Y_masked = X[mask].view(batch_size, n_cntxt, y_dim)

        return X_masked.contiguous(), Y_masked.contiguous()


class SuperresolutionCntxtTrgtGetter(GridCntxtTrgtGetter):
    """
    Split grids of values (e.g. images) into context and target points for super resolution.

    Parameters
    ----------
    resolution_factor : float, optional 
        Factor by which to change the resolution of context set. 

    downsample_mode : {"nearest","area","bilinear"}, optional
        Way of downsampling the image.

    **kwargs :
        Additional arguments to `GridCntxtTrgtGetter`.
    """

    def __init__(self, resolution_factor=1 / 4, downsample_mode="area", **kwargs):
        self.resolution_factor = resolution_factor
        self.downsample_mode = downsample_mode
        super().__init__(
            context_masker=ResolutionMasker(factor=int(1 / self.resolution_factor)),
            target_masker=no_masker,
            **kwargs
        )

    def preprocess_context(self, X):
        """Preprocess the data for the context set."""
        X = channels_to_2nd_dim(X)
        X_downscale = F.interpolate(
            X, scale_factor=self.resolution_factor, mode=self.downsample_mode
        )
        # upscale but with "nearest neighbor" interpolation => keep low resolution
        X_lowres = F.interpolate(
            X_downscale, scale_factor=int(1 / self.resolution_factor), mode="nearest"
        )
        return channels_to_last_dim(X_lowres)
