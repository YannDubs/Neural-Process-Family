"""Module for base of [conditional | latent] neural processes"""
import abc
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from npf.architectures import MLP, merge_flat_input
from npf.utils.helpers import (
    MultivariateNormalDiag,
    isin_range,
)
from npf.utils.initialization import weights_init

from .helpers import pool_and_replicate_middle

__all__ = [
    "NeuralProcessFamily",
    "LatentNeuralProcessFamily",
]


class NeuralProcessFamily(nn.Module, abc.ABC):
    """
    Base class for members of the neural process family.

    Notes
    -----
    - when writing size of vectors something like `size=[batch_size,*n_cntxt,y_dim]` means that the
    first dimension is the batch, the last is the target values and everything in the middle are context
    points. We use `*n_cntxt` as it can  be a single flattened dimension or many (for example on the grid).

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    encoded_path : {"latent", "both", "deterministic"}
        Which path(s) to use:
        - `"deterministic"` no latents : the decoder gets a deterministic representation s input.
        - `"latent"` uses latent : the decoder gets a sample latent representation as input.
        - `"both"` concatenates both the deterministic and sampled latents as input to the decoder.

    r_dim : int, optional
        Dimension of representations.

    x_transf_dim : int, optional
        Dimension of the encoded X. If `-1` uses `r_dim`. if `None` uses `x_dim`.

    is_heteroskedastic : bool, optional
        Whether the posterior predictive std can depend on the target features. If using in conjuction
        to `NllLNPF`, it might revert to a *CNP model (collapse of latents). If the flag is False, it
        pools all the scale parameters of the posterior distribution. This trick is only exactly
        recovers heteroskedasticity when the set target features are always the same (e.g.
        predicting values on a predefined grid) but is a good approximation even when not.

    XEncoder : nn.Module, optional
        Spatial encoder module which maps {x^i}_i -> {x_trnsf^i}_i. It should be
        constructable via `XEncoder(x_dim, x_transf_dim)`. `None` uses MLP. Example:
            - `MLP` : will learn positional embeddings with MLP
            - `SinusoidalEncodings` : use sinusoidal positional encodings.

    Decoder : nn.Module, optional
        Decoder module which maps {(x^t, r^t)}_t -> {p_y_suffstat^t}_t. It should be constructable
        via `decoder(x_dim, r_dim, n_out)`. If you have an decoder that maps
        [r;x] -> y you can convert it via `merge_flat_input(Decoder)`. `None` uses MLP. In the
        computational model this corresponds to `g`.
        Example:
            - `merge_flat_input(MLP)` : predict with MLP.
            - `merge_flat_input(SelfAttention, is_sum_merge=True)` : predict
            with self attention mechanisms (using `X_transf + Y` as input) to have
            coherent predictions (not use in attentive neural process [1] but in
            image transformer [2]).
            - `discard_ith_arg(MLP, 0)` if want the decoding to only depend on r.

    PredictiveDistribution : torch.distributions.Distribution, optional
        Predictive distribution. The input to the constructor are currently two values of the same
        shape : `loc` and `scale`, that are preprocessed by `p_y_loc_transformer` and
        `pred_scale_transformer`.

    p_y_loc_transformer : callable, optional
        Transformation to apply to the predicted location (e.g. mean for Gaussian)
        of Y_trgt.

    p_y_scale_transformer : callable, optional
        Transformation to apply to the predicted scale (e.g. std for Gaussian) of
        Y_trgt. The default follows [3] by using a minimum of 0.01.

    References
    ----------
    [1] Kim, Hyunjik, et al. "Attentive neural processes." arXiv preprint
        arXiv:1901.05761 (2019).
    [2] Parmar, Niki, et al. "Image transformer." arXiv preprint arXiv:1802.05751
        (2018).
    [3] Le, Tuan Anh, et al. "Empirical Evaluation of Neural Process Objectives."
        NeurIPS workshop on Bayesian Deep Learning. 2018.
    """

    _valid_paths = ["deterministic", "latent", "both"]

    def __init__(
        self,
        x_dim,
        y_dim,
        encoded_path,
        r_dim=128,
        x_transf_dim=-1,
        is_heteroskedastic=True,
        XEncoder=None,
        Decoder=None,
        PredictiveDistribution=MultivariateNormalDiag,
        p_y_loc_transformer=nn.Identity(),
        p_y_scale_transformer=lambda y_scale: 0.01 + 0.99 * F.softplus(y_scale),
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.encoded_path = encoded_path
        self.is_heteroskedastic = is_heteroskedastic

        if x_transf_dim is None:
            self.x_transf_dim = self.x_dim
        elif x_transf_dim == -1:
            self.x_transf_dim = self.r_dim
        else:
            self.x_transf_dim = x_transf_dim

        self.encoded_path = encoded_path.lower()
        if self.encoded_path not in self._valid_paths:
            raise ValueError(f"Unknown encoded_path={self.encoded_path}.")

        if XEncoder is None:
            XEncoder = self.dflt_Modules["XEncoder"]

        if Decoder is None:
            Decoder = self.dflt_Modules["Decoder"]

        self.x_encoder = XEncoder(self.x_dim, self.x_transf_dim)

        # times 2 out because loc and scale (mean and var for gaussian)
        self.decoder = Decoder(self.x_transf_dim, self.r_dim, self.y_dim * 2)

        self.PredictiveDistribution = PredictiveDistribution
        self.p_y_loc_transformer = p_y_loc_transformer
        self.p_y_scale_transformer = p_y_scale_transformer

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    @property
    def dflt_Modules(self):
        dflt_Modules = dict()

        dflt_Modules["XEncoder"] = partial(
            MLP, n_hidden_layers=1, hidden_size=self.r_dim
        )

        dflt_Modules["SubDecoder"] = partial(
            MLP,
            n_hidden_layers=4,
            hidden_size=self.r_dim,
        )

        dflt_Modules["Decoder"] = merge_flat_input(
            dflt_Modules["SubDecoder"], is_sum_merge=True
        )

        return dflt_Modules

    def forward(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None):
        """
        Given a set of context feature-values {(x^c, y^c)}_c and target features {x^t}_t, return
        a set of posterior distribution for target values {p(Y^t|y_c; x_c, x_t)}_t.

        Parameters
        ----------
        X_cntxt: torch.Tensor, size=[batch_size, *n_cntxt, x_dim]
            Set of all context features {x_i}. Values need to be in interval [-1,1]^d.

        Y_cntxt: torch.Tensor, size=[batch_size, *n_cntxt, y_dim]
            Set of all context values {y_i}.

        X_trgt: torch.Tensor, size=[batch_size, *n_trgt, x_dim]
            Set of all target features {x_t}. Values need to be in interval [-1,1]^d.

        Y_trgt: torch.Tensor, size=[batch_size, *n_trgt, y_dim], optional
            Set of all target values {y_t}. Only required during training and if
            using latent path.

        Return
        ------
        p_y_trgt: torch.distributions.Distribution, batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
            Posterior distribution for target values {p(Y^t|y_c; x_c, x_t)}_t

        z_samples: torch.Tensor, size=[n_z_samples, batch_size, *n_lat, r_dim]
            Sampled latents. `None` if `encoded_path==deterministic`.

        q_zCc: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[r_dim]
            Latent distribution for the context points. `None` if `encoded_path==deterministic`.

        q_zCct: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[r_dim]
            Latent distribution for the targets. `None` if `encoded_path==deterministic`
            or not training or not `is_q_zCct`.
        """
        self._validate_inputs(X_cntxt, Y_cntxt, X_trgt, Y_trgt)

        # size = [batch_size, *n_cntxt, x_transf_dim]
        X_cntxt = self.x_encoder(X_cntxt)
        # size = [batch_size, *n_trgt, x_transf_dim]
        X_trgt = self.x_encoder(X_trgt)

        # {R^u}_u
        # size = [batch_size, *n_rep, r_dim]
        R = self.encode_globally(X_cntxt, Y_cntxt)

        if self.encoded_path in ["latent", "both"]:
            z_samples, q_zCc, q_zCct = self.latent_path(X_cntxt, R, X_trgt, Y_trgt)
        else:
            z_samples, q_zCc, q_zCct = None, None, None

        if self.encoded_path == "latent":
            # if only latent path then cannot depend on deterministic representation
            R = None

        # size = [n_z_samples, batch_size, *n_trgt, r_dim]
        R_trgt = self.trgt_dependent_representation(X_cntxt, z_samples, R, X_trgt)

        # p(y|cntxt,trgt)
        # batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
        p_yCc = self.decode(X_trgt, R_trgt)

        return p_yCc, z_samples, q_zCc, q_zCct

    def _validate_inputs(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt):
        """Validates the inputs by checking if features are rescaled to [-1,1] during training."""
        if self.training:
            if not (isin_range(X_cntxt, [-1, 1]) and isin_range(X_trgt, [-1, 1])):
                raise ValueError(
                    f"Features during training should be in [-1,1]. {X_cntxt.min()} <= X_cntxt <= {X_cntxt.max()} ; {X_trgt.min()} <= X_trgt <= {X_trgt.max()}."
                )

    @abc.abstractmethod
    def encode_globally(self, X_cntxt, R_cntxt):
        """Encode context set all together (globally).

        Parameters
        ----------
        X_cntxt : torch.Tensor, size=[batch_size, *n_cntxt, x_transf_dim]
            Set of all context features {x^c}_c.

        Y_cntxt: torch.Tensor, size=[batch_size, *n_cntxt, y_dim]
            Set of all context values {y^c}_c.

        Return
        ------
        R : torch.Tensor, size=[batch_size, *n_rep, r_dim]
            Global representations of the context set.
        """
        pass

    @abc.abstractmethod
    def trgt_dependent_representation(self, X_cntxt, z_samples, R, X_trgt):
        """Compute a target dependent representation of the context set.

        Parameters
        ----------
        X_cntxt : torch.Tensor, size=[batch_size, *n_cntxt, x_transf_dim]
            Set of all context features {x^c}_c.

        z_samples: torch.Tensor, size=[n_z_samples, batch_size, *n_lat, r_dim]
            Sampled latents. `None` if `encoded_path==deterministic`.

        R : torch.Tensor, size=[batch_size, *n_rep, r_dim]
            Global representation of the context set. `None` if `self.encoded_path==latent`.

        X_trgt : torch.Tensor, size=[batch_size, *n_trgt, x_transf_dim]
            Set of all target features {x^t}_t.

        Returns
        -------
        R_trgt : torch.Tensor, size=[n_z_samples, batch_size, *n_trgt, r_dim]
            Set of all target representations {r^t}_t.
        """
        pass

    def latent_path(self, X_cntxt, R, X_trgt, Y_trgt):
        """Infer latent variable given context features and global representation.

        Parameters
        ----------
        R : torch.Tensor, size=[batch_size, *n_rep, r_dim]
            Global representation values {r^u}_u.

        X_cntxt : torch.Tensor, size=[batch_size, *n_cntxt, x_transf_dim]
            Set of all context features {x^c}_c.

        X_trgt : torch.Tensor, size=[batch_size, *n_trgt, x_transf_dim]
            Set of all target features {x^t}_t.

        Y_trgt: torch.Tensor, size=[batch_size, *n_trgt, y_dim], optional
            Set of all target values {y_t}. Only required during training and if
            using latent path.

        Return
        ------
        z_samples: torch.Tensor, size=[n_z_samples, batch_size, *n_lat, r_dim]
            Sampled latents. `None` if `encoded_path==deterministic`.

        q_zCc: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[r_dim]
            Latent distribution for the context points. `None` if `encoded_path==deterministic`.

        q_zCct: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[r_dim]
            Latent distribution for the targets. `None` if `encoded_path==deterministic`
            or not training or not `is_q_zCct`.
        """
        raise NotImplementedError(
            f"`latent_path` not implemented. Cannot use encoded_path={self.encoded_path} in such case."
        )

    def decode(self, X_trgt, R_trgt):
        """
        Compute predicted distribution conditioned on representation and
        target positions.

        Parameters
        ----------
        X_trgt: torch.Tensor, size=[batch_size, *n_trgt, x_transf_dim]
            Set of all target features {x^t}_t.

        R_trgt : torch.Tensor, size=[n_z_samples, batch_size, *n_trgt, r_dim]
            Set of all target representations {r^t}_t.

        Return
        ------
        p_y_trgt: torch.distributions.Distribution, batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
            Posterior distribution for target values {p(Y^t|y_c; x_c, x_t)}_t

        """
        # size = [n_z_samples, batch_size, *n_trgt, y_dim*2]
        p_y_suffstat = self.decoder(X_trgt, R_trgt)

        # size = [n_z_samples, batch_size, *n_trgt, y_dim]
        p_y_loc, p_y_scale = p_y_suffstat.split(self.y_dim, dim=-1)

        p_y_loc = self.p_y_loc_transformer(p_y_loc)
        p_y_scale = self.p_y_scale_transformer(p_y_scale)

        #! shuld probably pool before p_y_scale_transformer
        if not self.is_heteroskedastic:
            # to make sure not heteroskedastic you pool all the p_y_scale
            # only exact when X_trgt is a constant (e.g. grid case). If not it's a descent approx
            n_z_samples, batch_size, *n_trgt, y_dim = p_y_scale.shape
            p_y_scale = p_y_scale.view(n_z_samples * batch_size, *n_trgt, y_dim)
            p_y_scale = pool_and_replicate_middle(p_y_scale)
            p_y_scale = p_y_scale.view(n_z_samples, batch_size, *n_trgt, y_dim)

        # batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
        p_yCc = self.PredictiveDistribution(p_y_loc, p_y_scale)

        return p_yCc

    def set_extrapolation(self, min_max):
        """Set the neural process for extrapolation."""
        pass


class LatentNeuralProcessFamily(NeuralProcessFamily):
    """Base class for members of the latent neural process (sub-)family.

    Parameters
    ----------
    *args:
        Positional arguments to `NeuralProcessFamily`.

    encoded_path : {"latent", "both"}
        Which path(s) to use:
        - `"latent"` uses latent : the decoder gets a sample latent representation as input.
        - `"both"` concatenates both the deterministic and sampled latents as input to the decoder.

    is_q_zCct : bool, optional
        Whether to infer Z using q(Z|cntxt,trgt) instead of q(Z|cntxt). This requires the loss
        to perform some type of importance sampling. Only used if `encoded_path in {"latent", "both"}`.

    n_z_samples_train : int or scipy.stats.rv_frozen, optional
        Number of samples from the latent during training. Only used if `encoded_path in {"latent", "both"}`.
        Can also be a scipy random variable , which is useful if the number of samples has to be stochastic, for
        example when using `SUMOLossNPF`.

    n_z_samples_test : int or scipy.stats.rv_frozen, optional
        Number of samples from the latent during testing. Only used if `encoded_path in {"latent", "both"}`.
        Can also be a scipy random variable , which is useful if the number of samples has to be stochastic, for
        example when using `SUMOLossNPF`.

    LatentEncoder : nn.Module, optional
        Encoder which maps r -> z_suffstat. It should be constructed via
        `LatentEncoder(r_dim, n_out)`.  If `None` uses an MLP.

    LatentDistribution : torch.distributions.Distribution, optional
        Latent distribution. The input to the constructor are currently two values  : `loc` and `scale`,
        that are preprocessed by `q_z_loc_transformer` and `q_z_loc_transformer`.

    q_z_loc_transformer : callable, optional
        Transformation to apply to the predicted location (e.g. mean for Gaussian)
        of Y_trgt.

    q_z_scale_transformer : callable, optional
        Transformation to apply to the predicted scale (e.g. std for Gaussian) of
        Y_trgt. The default follows [3] by using a minimum of 0.1 and maximum of 1.

    **kwargs:
        Additional arguments to `NeuralProcessFamily`.
    """

    _valid_paths = ["latent", "both"]

    def __init__(
        self,
        *args,
        is_q_zCct=False,
        n_z_samples_train=32,
        n_z_samples_test=32,
        LatentEncoder=None,
        LatentDistribution=MultivariateNormalDiag,
        q_z_loc_transformer=nn.Identity(),
        q_z_scale_transformer=lambda z_scale: 0.1 + 0.9 * torch.sigmoid(z_scale),
        z_dim=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.is_q_zCct = is_q_zCct
        self.n_z_samples_train = n_z_samples_train
        self.n_z_samples_test = n_z_samples_test
        self.z_dim = self.r_dim if z_dim is None else z_dim

        if LatentEncoder is None:
            LatentEncoder = self.dflt_Modules["LatentEncoder"]

        # times 2 out because loc and scale (mean and var for gaussian)
        self.latent_encoder = LatentEncoder(self.r_dim, self.z_dim * 2)

        if self.encoded_path == "both":
            self.r_z_merger = nn.Linear(self.r_dim + self.z_dim, self.r_dim)

        self.LatentDistribution = LatentDistribution
        self.q_z_loc_transformer = q_z_loc_transformer
        self.q_z_scale_transformer = q_z_scale_transformer

        if self.z_dim != self.r_dim and self.encoded_path == "latent":
            # will reshape the z samples to make sure they can be given to the decoder
            self.reshaper_z = nn.Linear(self.z_dim, self.r_dim)

        self.reset_parameters()

    @property
    def dflt_Modules(self):
        # allow inheritence
        dflt_Modules = NeuralProcessFamily.dflt_Modules.__get__(self)

        dflt_Modules["LatentEncoder"] = partial(
            MLP,
            n_hidden_layers=1,
            hidden_size=self.r_dim,
        )

        return dflt_Modules

    def forward(self, *args, **kwargs):

        # make sure that only sampling oce per loop => cannot be a property
        try:
            # if scipy random variable, i.e., random number of samples
            self.n_z_samples = (
                self.n_z_samples_train.rvs()
                if self.training
                else self.n_z_samples_test.rvs()
            )
        except AttributeError:
            self.n_z_samples = (
                self.n_z_samples_train if self.training else self.n_z_samples_test
            )

        return super().forward(*args, **kwargs)

    def _validate_inputs(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt):
        super()._validate_inputs(X_cntxt, Y_cntxt, X_trgt, Y_trgt)

    def latent_path(self, X_cntxt, R, X_trgt, Y_trgt):

        # q(z|c)
        # batch shape = [batch_size, *n_lat] ; event shape = [z_dim]
        q_zCc = self.infer_latent_dist(X_cntxt, R)

        if self.is_q_zCct and Y_trgt is not None:
            # during training when we know Y_trgt, we can take an expectation over q(z|cntxt,trgt)
            # instead of q(z|cntxt). note that actually does q(z|trgt) because trgt has cntxt
            R_from_trgt = self.encode_globally(X_trgt, Y_trgt)
            q_zCct = self.infer_latent_dist(X_trgt, R_from_trgt)
            sampling_dist = q_zCct
        else:
            q_zCct = None
            sampling_dist = q_zCc

        # size = [n_z_samples, batch_size, *n_lat, z_dim]
        z_samples = sampling_dist.rsample([self.n_z_samples])

        return z_samples, q_zCc, q_zCct

    def infer_latent_dist(self, X, R):
        """Infer latent distribution given desired features and global representation.

        Parameters
        ----------
        X : torch.Tensor, size=[batch_size, *n_i, x_transf_dim]
            Set of all features {x^i}_i. E.g. context or target.

        R : torch.Tensor, size=[batch_size, *n_rep, r_dim]
            Global representation values {r^u}_u.

        Return
        ------
        q_zCc: torch.distributions.Distribution, batch shape = [batch_size, *n_lat] ; event shape = [z_dim]
            Inferred latent distribution.
        """

        # size = [batch_size, *n_lat, z_dim]
        R_lat_inp = self.rep_to_lat_input(R)

        # size = [batch_size, *n_lat, z_dim*2]
        q_z_suffstat = self.latent_encoder(R_lat_inp)

        q_z_loc, q_z_scale = q_z_suffstat.split(self.z_dim, dim=-1)

        q_z_loc = self.q_z_loc_transformer(q_z_loc)
        q_z_scale = self.q_z_scale_transformer(q_z_scale)

        # batch shape = [batch_size, *n_lat] ; event shape = [z_dim]
        q_zCc = self.LatentDistribution(q_z_loc, q_z_scale)

        return q_zCc

    def rep_to_lat_input(self, R):
        """Transform the n_rep representations to n_lat inputs."""
        # by default *n_rep = *n_lat
        return R

    def merge_r_z(self, R, z_samples):
        """
        Merges the deterministic representation and sampled latent. Assumes that n_lat = n_rep.

        Parameters
        ----------
        R : torch.Tensor, size=[batch_size, *, r_dim]
            Global representation values {r^u}_u.

        z_samples : torch.Tensor, size=[n_z_samples, batch_size, *, r_dim]
            Global representation values {r^u}_u.

        Return
        ------
        out : torch.Tensor, size=[n_z_samples, batch_size, *, r_dim]
        """
        if R.shape != z_samples.shape:

            R = R.unsqueeze(0).expand(*z_samples.shape[:-1], self.r_dim)

        # (add ReLU to not have linear followed by linear)
        return torch.relu(self.r_z_merger(torch.cat((R, z_samples), dim=-1)))
