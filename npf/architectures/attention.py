import abc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.distance import CosineSimilarity, PairwiseDistance

from npf.utils.initialization import weights_init

from .mlp import MLP

__all__ = ["get_attender"]


def get_attender(attention, kq_size, value_size, out_size, **kwargs):
    """
    Set scorer that matches key and query to compute attention along `dim=1`.

    Parameters
    ----------
    attention: callable or {'multiplicative', "additive", "scaledot", "multihead",
            "manhattan", "euclidean", "cosine", "transformer", "weighted_dist"}, optional
        The method to compute the alignment. If not a string (callable) will return
        it. Else: `"scaledot"` mitigates the high dimensional issue of the scaled
        product by rescaling it [1]. `"multihead"` is the same with multiple heads
        [1]. `"transformer"` builds upon `"multihead"` by adding layer normalization
        and skip connction as described in [2]. "additive"` is the original attention
        [3]. `"multiplicative"` is  faster and more space efficient [4] but performs
        a little bit worst for high dimensions. `"cosine"` cosine similarity.
        `"manhattan"` `"euclidean"` are the negative distances and "weighted_dist"
        is the negative distance with different dimension weights.

    kq_size : int
        Size of the key and query.

    value_size : int
        Final size of the value.

    out_size : int
        Output dimension.

    kwargs :
        Additional arguments to the attender.

    References
    ----------
    [1] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural
        information processing systems. 2017.
    [2] Parmar, Niki, et al. "Image transformer." arXiv preprint arXiv:1802.05751
        (2018).
    [2] Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine
        translation by jointly learning to align and translate." arXiv preprint
        arXiv:1409.0473 (2014).
    [3] Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning. "Effective
        approaches to attention-based neural machine translation." arXiv preprint
        arXiv:1508.04025 (2015).
    """
    if not isinstance(attention, str):
        return attention(kq_size, value_size, out_size, **kwargs)

    attention = attention.lower()
    if attention == "multiplicative":
        attender = MultiplicativeAttender(kq_size, value_size, out_size, **kwargs)
    elif attention == "additive":
        attender = AdditiveAttender(kq_size, value_size, out_size, **kwargs)
    elif attention == "scaledot":
        attender = DotAttender(kq_size, value_size, out_size, is_scale=True, **kwargs)
    elif attention == "cosine":
        attender = CosineAttender(kq_size, value_size, out_size, **kwargs)
    elif attention == "manhattan":
        attender = DistanceAttender(kq_size, value_size, out_size, p=1, **kwargs)
    elif attention == "euclidean":
        attender = DistanceAttender(kq_size, value_size, out_size, p=2, **kwargs)
    elif attention == "weighted_dist":
        attender = DistanceAttender(
            kq_size, value_size, out_size, is_weight=True, p=1, **kwargs
        )
    elif attention == "multihead":
        attender = MultiheadAttender(kq_size, value_size, out_size, **kwargs)
    elif attention == "transformer":
        attender = TransformerAttender(kq_size, value_size, out_size, **kwargs)
    else:
        raise ValueError("Unknown attention method {}".format(attention))

    return attender


class BaseAttender(abc.ABC, nn.Module):
    """
    Base Attender module.

    Parameters
    ----------
    kq_size : int
        Size of the key and query.

    value_size : int
        Final size of the value.

    out_size : int
        Output dimension. If not different than `kq_size` will do all the computation
        with a size of `x_dim` and add a linear layer at the end to reshape.

    is_normalize : bool, optional
        Whether weights should sum to 1 (using softmax).

    dropout : float, optional
        Dropout rate to apply to the attention.
    """

    def __init__(self, kq_size, value_size, out_size, is_normalize=True, dropout=0):
        super().__init__()
        self.kq_size = kq_size
        self.value_size = value_size
        self.out_size = out_size
        self.is_normalize = is_normalize
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.is_resize = self.value_size != self.out_size

        if self.is_resize:
            self.resizer = nn.Linear(self.value_size, self.out_size)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, keys, queries, values, **kwargs):
        """
        Compute the attention between given key and queries.

        Parameters
        ----------
        keys : torch.Tensor, size=[batch_size, n_keys, kq_size]
        queries : torch.Tensor, size=[batch_size, n_queries, kq_size]
        values : torch.Tensor, size=[batch_size, n_keys, value_size]

        Return
        ------
        context : torch.Tensor, size=[batch_size, n_queries, out_size]
        """
        logits = self.score(keys, queries, **kwargs)

        attn = self.logits_to_attn(logits)

        attn = self.dropout(attn)

        # attn : size=[batch_size, n_queries, n_keys]
        # values : size=[batch_size, n_keys, value_size]
        context = torch.bmm(attn, values)

        if self.is_resize:
            context = self.resizer(context)

        return context

    def logits_to_attn(self, logits):
        """Convert logits to attention."""
        if self.is_normalize:
            attn = logits.softmax(dim=-1)
        else:
            attn = logits
        return attn

    @abc.abstractmethod
    def score(keys, queries, **kwargs):
        """Score function which returns the logits between keys and queries."""
        pass


class DotAttender(BaseAttender):
    """
    Dot product attention.

    Parameters
    ----------
    kq_size : int
        Size of the key and query.

    value_size : int
        Final size of the value.

    out_size : int
        Output dimension.

    is_scale: bool, optional
        whether to use a scaled attention just like in [1]. Scaling can help when
        dimension is large by making sure that there are no extremely small gradients.

    kwargs:
        Additional arguments to `BaseAttender`.

    References
    ----------
    [1] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural
        information processing systems. 2017.
    """

    def __init__(self, *args, is_scale=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_scale = is_scale

    def score(self, keys, queries):
        # b: batch_size, q: n_queries, k: n_keys, d: kq_size
        # e.g. if keys have 4 dimension it means that different queries will
        # be associated with different keys
        keys_shape = "bqkd" if len(keys.shape) == 4 else "bkd"
        queries_shape = "bqkd" if len(queries.shape) == 4 else "bqd"

        # [batch_size, n_queries, kq_size]
        logits = torch.einsum(
            "{},{}->bqk".format(keys_shape, queries_shape), keys, queries
        )

        if self.is_scale:
            kq_size = queries.size(-1)
            logits = logits / math.sqrt(kq_size)

        return logits


class MultiplicativeAttender(BaseAttender):
    """
    Multiplicative attention mechanism [1].

    Parameters
    ----------
    kq_size : int
        Size of the key and query.

    value_size : int
        Final size of the value.

    out_size : int
        Output dimension.

    kwargs:
        Additional arguments to `BaseAttender`.

    References
    ----------
    [1] Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning. "Effective
        approaches to attention-based neural machine translation." arXiv preprint
        arXiv:1508.04025 (2015).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.linear = nn.Linear(self.kq_size, self.kq_size, bias=False)
        self.dot = DotAttender(*args, is_scale=False)
        self.reset_parameters()

    def score(self, keys, queries):
        transformed_queries = self.linear(queries)
        logits = self.dot.score(keys, transformed_queries)
        return logits


class AdditiveAttender(BaseAttender):
    """
    Original additive attention mechanism [1].

    Parameters
    ----------
    kq_size : int
        Size of the key and query.

    value_size : int
        Final size of the value.

    out_size : int
        Output dimension.

    kwargs:
        Additional arguments to `BaseAttender`.

    References
    ----------
    [1] Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine
        translation by jointly learning to align and translate." arXiv preprint
        arXiv:1409.0473 (2014).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mlp = MLP(
            self.kq_size * 2, 1, hidden_size=self.kq_size, activation=nn.Tanh()
        )
        self.reset_parameters()

    def score(self, keys, queries):
        batch_size, n_queries, kq_size = queries.shape
        n_keys = keys.size(1)

        keys = keys.unsqueeze(1).expand(batch_size, n_queries, n_keys, kq_size)
        queries = queries.unsqueeze(1).expand(batch_size, n_queries, n_keys, kq_size)

        logits = self.mlp(torch.cat((keys, queries), dim=-1)).squeeze(-1)
        return logits


class CosineAttender(BaseAttender):
    """
    Computes the attention as a function of cosine similarity.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity = CosineSimilarity(dim=1)

    def score(self, keys, queries):
        batch_size, n_queries, kq_size = queries.size()
        n_keys = keys.size(1)

        keys = keys.view(batch_size, kq_size, 1, n_keys)
        queries = queries.view(batch_size, kq_size, n_queries, 1)
        logits = self.similarity(keys, queries)

        return logits


class DistanceAttender(BaseAttender):
    """
    Computes the attention as a function of the negative dimension wise (weighted)
    distance.

    Parameters
    ----------
    kq_size : int
        Size of the key and query.

    value_size : int
        Final size of the value.

    out_size : int
        Output dimension.

    p : float, optional
        The exponent value in the norm formulation.

    is_weight : float, optional
        Whether to use a dimension wise weight and bias.

    kwargs :
        Additional arguments to `BaseAttender`.
    """

    def __init__(self, *args, p=1, is_weight=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.is_weight = is_weight
        if self.is_weight:
            self.weighter = nn.Linear(self.kq_size, self.kq_size)

        self.reset_parameters()

    def score(self, keys, queries):
        batch_size, n_queries, kq_size = queries.size()
        n_keys = keys.size(1)

        keys = keys.view(batch_size, 1, n_keys, kq_size)
        queries = queries.view(batch_size, n_queries, 1, kq_size)
        diff = keys - queries
        if self.is_weight:
            diff = self.weighter(diff)

        logits = -torch.norm(diff, p=self.p, dim=-1) ** 2

        return logits


class MultiheadAttender(nn.Module):
    """
    Multihead attention mechanism [1].

    Parameters
    ----------
    kq_size : int
        Size of the key and query. Needs to be a multiple of `n_heads`.

    value_size : int
        Final size of the value. Needs to be a multiple of `n_heads`.

    out_size : int
        Output dimension.

    n_heads : int, optional
        Number of heads

    is_post_process : bool, optional
        Whether to pos process the outout with a linear layer.

    dropout : float, optional
        Dropout rate to apply to the attention.

    is_relative_pos : bool, optional
        Whether to add some relative position encodings. If `True` the positional
        encoding of size `kq_size` should be given in the `forward pass`.

    References
    ----------
    [1] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural
        information processing systems. 2017.
    """

    def __init__(
        self,
        kq_size,
        value_size,
        out_size,
        n_heads=8,
        is_post_process=True,
        dropout=0,
        is_relative_pos=False,
    ):
        super().__init__()
        self.is_relative_pos = is_relative_pos
        # only 3 transforms for scalability but actually as if using n_heads * 3 layers
        self.key_transform = nn.Linear(kq_size, kq_size, bias=False)
        self.query_transform = nn.Linear(
            kq_size, kq_size, bias=not self.is_relative_pos
        )
        self.value_transform = nn.Linear(value_size, value_size, bias=False)
        self.dot = DotAttender(
            kq_size, value_size, out_size, is_scale=True, dropout=dropout
        )
        self.n_heads = n_heads
        self.kq_head_size = kq_size // self.n_heads
        self.value_head_size = kq_size // self.n_heads
        self.kq_size = kq_size
        self.value_size = value_size
        self.out_size = out_size
        self.post_processor = (
            nn.Linear(value_size, out_size)
            if is_post_process or value_size != out_size
            else None
        )

        assert kq_size % n_heads == 0, "{} % {} != 0".format(kq_size, n_heads)
        assert value_size % n_heads == 0, "{} % {} != 0".format(value_size, n_heads)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

        # change initialization because real output is not kqv_size but head_size
        # just coded so for convenience and scalability
        std = math.sqrt(2.0 / (self.kq_size + self.kq_head_size))
        nn.init.normal_(self.key_transform.weight, mean=0, std=std)
        nn.init.normal_(self.query_transform.weight, mean=0, std=std)
        std = math.sqrt(2.0 / (self.value_size + self.value_head_size))
        nn.init.normal_(self.value_transform.weight, mean=0, std=std)

    def forward(self, keys, queries, values, rel_pos_enc=None, **kwargs):
        """
        Compute the attention between given key and queries.

        Parameters
        ----------
        keys: torch.Tensor, size=[batch_size, n_keys, kq_size]
        queries: torch.Tensor, size=[batch_size, n_queries, kq_size]
        values: torch.Tensor, size=[batch_size, n_keys, value_size]
        rel_pos_enc: torch.Tensor, size=[batch_size, n_queries, n_keys, kq_size]
            Positional encoding with the differences between every key and query.

        Return
        ------
        context : torch.Tensor, size=[batch_size, n_queries, out_size]
        """
        keys = self.key_transform(keys)
        queries = self.query_transform(queries)
        values = self.value_transform(values)

        # Make multihead. Size = [batch_size * n_heads, {n_keys, n_queries}, head_size]
        queries = self._make_multiheaded(queries, self.kq_head_size)
        values = self._make_multiheaded(values, self.value_head_size)

        # keys have to add relative position before splitting head
        if self.is_relative_pos:
            # when relative position, every query has different associated key
            batch_size, n_keys, kq_size = keys.shape
            n_queries = queries.size(1)
            keys = (keys.unsqueeze(1) + rel_pos_enc).view(
                batch_size, n_queries * n_keys, kq_size
            )
            keys = self._make_multiheaded(keys, self.kq_head_size)
            keys = keys.view(
                batch_size * self.n_heads, n_queries, n_keys, self.kq_head_size
            )
        else:
            keys = self._make_multiheaded(keys, self.kq_head_size)

        # Size = [batch_size * n_heads, n_queries, head_size]
        context = self.dot(keys, queries, values)

        # Size = [batch_size, n_queries, value_size]
        context = self._concatenate_multiheads(context, self.value_head_size)

        if self.post_processor is not None:
            context = self.post_processor(context)

        return context

    def _make_multiheaded(self, kvq, head_size):
        """Make a key, value, query multiheaded by stacking the heads as new batches."""
        batch_size = kvq.size(0)
        kvq = kvq.view(batch_size, -1, self.n_heads, head_size)
        kvq = (
            kvq.permute(2, 0, 1, 3)
            .contiguous()
            .view(batch_size * self.n_heads, -1, head_size)
        )
        return kvq

    def _concatenate_multiheads(self, kvq, head_size):
        """Reverts `_make_multiheaded` by concatenating the heads."""
        batch_size = kvq.size(0) // self.n_heads
        kvq = kvq.view(self.n_heads, batch_size, -1, head_size)
        kvq = (
            kvq.permute(1, 2, 0, 3)
            .contiguous()
            .view(batch_size, -1, self.n_heads * head_size)
        )
        return kvq


class TransformerAttender(MultiheadAttender):
    """
    Image Transformer attention mechanism [1].

    Parameters
    ----------
    kq_size : int
        Size of the key and query. Needs to be a multiple of `n_heads`.

    value_size : int
        Final size of the value. Needs to be a multiple of `n_heads`.

    out_size : int
        Output dimension. Has to be the same size as `kq_size` due to the residual
        connection.

    kwargs:
        Additional arguments to `MultiheadAttender`.

    References
    ----------
    [1] Parmar, Niki, et al. "Image transformer." arXiv preprint arXiv:1802.05751
        (2018).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, is_post_process=False, **kwargs)
        assert self.kq_size == self.out_size
        self.layer_norm1 = nn.LayerNorm(self.out_size)
        self.layer_norm2 = nn.LayerNorm(self.out_size)
        self.mlp = MLP(
            self.out_size,
            self.out_size,
            hidden_size=self.out_size,
            activation=nn.ReLU(),
        )

        self.reset_parameters()

    def forward(self, keys, queries, values, **kwargs):
        """
        Compute the attention between given key and queries.

        Parameters
        ----------
        keys: torch.Tensor, size=[batch_size, n_keys, kqv_size]
        queries: torch.Tensor, size=[batch_size, n_queries, kqv_size]
        values: torch.Tensor, size=[batch_size, n_keys, kqv_size]

        Return
        ------
        context : torch.Tensor, size=[batch_size, n_queries, kqv_size]
        """
        context = super().forward(keys, queries, values, **kwargs)
        # residual connection + layer norm
        context = self.layer_norm1(context + queries)
        context = self.layer_norm2(context + self.dot.dropout(self.mlp(context)))

        return context
