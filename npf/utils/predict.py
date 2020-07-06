import torch

__all__ = [
    "VanillaPredictor",
    "AutoregressivePredictor",
    "GenNextAutoregressivePixelL1",
    "GenAllAutoregressivePixel",
    "SamplePredictor",
]


class SamplePredictor:
    """Multi sample prediction using a trained `NeuralProcess` model. `is_dist` if should return the 
    predictive instead of loc (e.g. mean)."""

    def __init__(self, model, is_dist=False):
        self.model = model
        self.is_dist = is_dist

    def __call__(self, *args):
        # q_zCc.base_dist.scale.mean()
        p_y_pred, *_ = self.model(*args)

        if self.is_dist:
            return p_y_pred

        loc_ys = p_y_pred.base_dist.loc.detach()
        return loc_ys


class VanillaPredictor(SamplePredictor):
    """Single shot prediction using a trained `NeuralProcess` model."""

    def __call__(self, *args):
        mean_ys = super().__call__(*args)
        return mean_ys[0]  # return the first z_sample


class GenAllAutoregressivePixel:
    """Helper function that returns all pixels at every step n times."""

    def __init__(self, n=100):
        self.n = n

    def __call__(self, mask_cntxt):
        next_mask_cntxt = mask_cntxt.clone()
        next_mask_cntxt[:] = 1

        for i in range(self.n):
            yield next_mask_cntxt.clone()


class GenNextAutoregressivePixelL1:
    """Generates the next autoregressive pixels by using the ones at `d` L1
    distance from context points.
    """

    def __init__(self, d):
        self.d = d

    def __call__(self, mask_cntxt):
        """
        Given the current context mask, return the next
        temporary target mask by setting all pixels than are at d manhattan distance
        of a context pixel.
        """
        next_mask_cntxt = mask_cntxt.clone()
        slcs = [slice(None)] * (len(mask_cntxt.shape))

        while not (next_mask_cntxt == 1).all():
            for _ in range(self.d):
                # shift array to the 4 directions to get all neighbors
                right, left, up, down = self.get_shifted_masks(next_mask_cntxt)
                next_mask_cntxt = right | left | up | down | next_mask_cntxt
            yield next_mask_cntxt.clone()

    def get_shifted_masks(self, mask):
        """Given a batch of masks , returns the same masks shifted to the right,
        left, up, down."""
        right_shifted = torch.cat(
            ((mask[:, :, -1:, ...] * 0).bool(), mask[:, :, :-1, ...]), dim=2
        )
        left_shifted = torch.cat(
            (mask[:, :, 1:, ...], (mask[:, :, :1, ...] * 0).bool()), dim=2
        )
        up_shifted = torch.cat(
            (mask[:, 1:, :, ...], (mask[:, :1, :, ...] * 0).bool()), dim=1
        )
        down_shifted = torch.cat(
            ((mask[:, -1:, :, ...] * 0).bool(), mask[:, :-1, :, ...]), dim=1
        )
        return right_shifted, left_shifted, up_shifted, down_shifted


class AutoregressivePredictor:
    """
    Autoregressive prediction using a trained `NeuralProcess` model.

    Parameters
    ----------
    model : nn.Module
        Model used to initialize `MeanPredictor`.

    gen_autoregressive_trgts : callable, optional
        Function which returns a generator of the next mask target given the initial
        mask context `get_next_tgrts(mask_cntxt)`.

    is_repredict : bool, optional
        Whether to is_repredict the given context and previous targets at
        each autoregressive temporary steps.
    """

    def __init__(
        self,
        model,
        gen_autoregressive_trgts=GenNextAutoregressivePixelL1(1),
        is_repredict=False,
    ):
        self.predictor = VanillaPredictor(model)
        self.gen_autoregressive_trgts = gen_autoregressive_trgts
        self.is_repredict = is_repredict

    def __call__(self, mask_cntxt, X, mask_trgt):
        X = X.clone()

        gen_cur_mask_trgt = self.gen_autoregressive_trgts(mask_cntxt)

        for cur_mask_trgt in gen_cur_mask_trgt:
            next_mask_cntxt = cur_mask_trgt.clone()

            if not self.is_repredict:
                # don't predict what is in cntxt
                cur_mask_trgt[mask_cntxt.squeeze(-1)] = 0

            mean_y = self.predictor(mask_cntxt, X, cur_mask_trgt)
            X[cur_mask_trgt.squeeze(-1)] = mean_y.view(-1, mean_y.shape[-1])
            mask_cntxt = next_mask_cntxt

        # predict once with all to have the actual trgt
        mean_y = self.predictor(mask_trgt, X, mask_trgt)

        return mean_y
