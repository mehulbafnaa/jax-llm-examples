# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal SmallThinker model definition."""

import dataclasses
import json
import math
import os
from functools import partial
from inspect import signature
from pathlib import Path
from typing import Any, Callable, TypeVar

import jax
import jax.numpy as jnp
from etils import epath
from jax import random, tree_util
from jax.experimental.pallas.ops.tpu.splash_attention import \
    splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import \
    splash_attention_mask as mask_lib
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
from jax.sharding import use_mesh

try:
    from jax.experimental.shard import auto_axes as _auto_axes
    from jax.experimental.shard import reshard
except ModuleNotFoundError:
    from jax.sharding import auto_axes as _auto_axes, reshard


from . import ragged_attention

AxisName = str | tuple[str, ...] | None
Axes = tuple[AxisName, ...] 

# Expected physical mesh axis names:
# x - batch
# y - 1st of 2D tensor sharding
# z - 2nd of 2D tensor sharding
BATCH_AXIS_NAME = "x"
PARTIAL_TENSOR_AXIS_NAME = "y"
TENSOR_AXIS_NAME = ("y", "z")


@dataclasses.dataclass
class ShardingRules:
    """Mapping from logical data axes to physical mesh axes."""

    batch: AxisName = BATCH_AXIS_NAME
    sequence: AxisName = None
    act_embed: AxisName = None
    act_heads: AxisName = None
    head_dim: AxisName = None
    # attention
    qkv_embed: AxisName = None
    q_heads: AxisName = TENSOR_AXIS_NAME
    kv_heads: AxisName = None  # Can't shard 2 KV heads across 8 devices
    o_heads: AxisName = TENSOR_AXIS_NAME
    o_embed: AxisName = None
    # MLP
    embed_up: AxisName = None
    ffw_up: AxisName = TENSOR_AXIS_NAME
    ffw_down: AxisName = TENSOR_AXIS_NAME
    embed_down: AxisName = None
    # vocab
    vocab_in: AxisName = None
    vocab_out: AxisName = TENSOR_AXIS_NAME
    # moe
    experts: AxisName = None


def auto_axes(x, out_sharding):
    argname = "out_sharding" if "out_sharding" in signature(_auto_axes).parameters else "out_shardings"
    return _auto_axes(x, **{argname: out_sharding})


def logical_to_physical(logical: Axes, rules: ShardingRules) -> jax.sharding.PartitionSpec:
    """Returns how to physically shard a given sequence of logical array dimensions."""
    spec = [getattr(rules, axis) if axis is not None else None for axis in logical]
    flat_axes = jax.tree.leaves(spec)
    if len(set(flat_axes)) != len(flat_axes):
        raise ValueError(f"Colliding physical axes from translating logical spec {logical} -> {spec}")
    return P(*spec)


def logical_to_sharding(logical: Axes, mesh: jax.sharding.Mesh, rules: ShardingRules) -> jax.sharding.Sharding:
    """Returns the sharding for a given sequence of logical array dimensions."""
    assert mesh is not None
    return jax.sharding.NamedSharding(mesh, logical_to_physical(logical, rules))


def jax_pytree_struct(cls, meta_fields: tuple = ()): 
    """jax.tree_util.register_dataclass wrapper that automatically infers data_fields."""
    if not dataclasses.is_dataclass(cls):
        cls = dataclasses.dataclass(cls)
    all_fields = tuple(f.name for f in dataclasses.fields(cls) if f.init)
    data_fields = tuple(f for f in all_fields if f not in meta_fields)
    return tree_util.register_dataclass(cls, data_fields=data_fields, meta_fields=meta_fields)


jax_static = lambda cls: tree_util.register_static(dataclasses.dataclass(cls))


@jax_static
class Config:
    embed: int
    ffw_size: int
    q_heads: int
    kv_heads: int
    num_layers: int
    head_dim: int
    vocab_size: int
    max_seq_len: int
    causal: bool
    use_prefill_attn_kernel: bool
    use_decode_attn_kernel: bool
    dtype: "jnp.dtype" = jnp.bfloat16
    # sharding
    rules: ShardingRules = dataclasses.field(default_factory=ShardingRules)
    mesh: jax.sharding.Mesh | None = None
    # SmallThinker specific
    rope_theta: float = 1.5e6
    sliding_window_size: int = 4096
    moe_num_experts: int = 32
    moe_num_experts_per_tok: int = 4
    # Quantization
    quant_layer: bool = True
    quant_cache: bool = True
    quant_scale_dtype: "jnp.dtype" = jnp.float16


def smallthinker_to_jax_config(smallthinker_config: Any | dict[str, Any]) -> "Config":
    _get = lambda x, k, default=None: getattr(x, k, default) if hasattr(x, k) else dict(x).get(k, default)
    return Config(
        embed=_get(smallthinker_config, "hidden_size"),
        ffw_size=_get(smallthinker_config, "moe_ffn_hidden_size"),
        q_heads=_get(smallthinker_config, "num_attention_heads"),
        kv_heads=_get(smallthinker_config, "num_key_value_heads"),
        num_layers=_get(smallthinker_config, "num_hidden_layers"),
        head_dim=_get(smallthinker_config, "head_dim"),
        vocab_size=_get(smallthinker_config, "vocab_size"),
        max_seq_len=128,
        dtype=jnp.bfloat16,
        causal=True,
        use_prefill_attn_kernel=False,
        use_decode_attn_kernel=False,
        rope_theta=_get(smallthinker_config, "rope_theta"),
        sliding_window_size=_get(smallthinker_config, "sliding_window_size"),
        moe_num_experts=_get(smallthinker_config, "moe_num_primary_experts"),
        moe_num_experts_per_tok=_get(smallthinker_config, "moe_num_active_primary_experts"),
    )


def load_config(config_path: str | os.PathLike[str] | Path) -> "Config":
    return smallthinker_to_jax_config(json.loads(Path(config_path).read_text()))


PreTrainedTokenizerFast = TypeVar("PreTrainedTokenizerFast")


def load_tokenizer(
    tokenizer_path: str | os.PathLike[str] | Path, tokenizer_config_path: str | os.PathLike[str] | Path
) -> PreTrainedTokenizerFast:
    from transformers import AddedToken, PreTrainedTokenizerFast

    config = json.loads(Path(tokenizer_config_path).read_text())
    config = {k: AddedToken(**v) if isinstance(v, dict) and str(k).endswith("token") else v for (k, v) in config.items()}
    config["added_tokens_decoder"] = {
        int(k): AddedToken(**v) for (k, v) in config.get("added_tokens_decoder", dict()).items()
    }
    return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path), **config)


@partial(jax_pytree_struct, meta_fields=("shape", "logical_axes", "initializer", "dtype"))
@dataclasses.dataclass
class ArrayInfo:
    shape: tuple[int, ...]
    dtype: "jnp.dtype"
    logical_axes: tuple
    initializer: Callable | None = None


is_type = lambda x, cls: (type(x).__name__ == cls.__name__) and (type(x).__module__ == cls.__module__)
is_param = lambda x: is_type(x, ArrayInfo)
_count_left_padding = lambda ids, pad_id=0: auto_axes(
    lambda ids: jnp.sum(jnp.cumsum(ids != pad_id, axis=-1) == 0, axis=-1), out_sharding=P(None)
)(ids)
_length_minus_padding = lambda segment_ids: auto_axes(
    lambda segment_ids: jnp.sum(jnp.cumsum(jnp.flip(segment_ids != 0, -1), axis=-1) > 0, -1), out_sharding=P(None)
)(segment_ids)


def _init_leaves(key, abstract, shardings):
    # Extract shapes and dtypes as hashable static args
    shapes_and_dtypes = tuple((info.shape, info.dtype) for info in abstract)
    
    @partial(jax.jit, static_argnames=("shapes_and_dtypes",), out_shardings=shardings)
    def _init_fn(key, shapes_and_dtypes):
        keys = random.split(key, len(shapes_and_dtypes))
        results = []
        for i, (k, (shape, dtype)) in enumerate(zip(keys, shapes_and_dtypes)):
            init_fn = abstract[i].initializer  # Get initializer from closure
            results.append(init_fn(k, shape, dtype))
        return tuple(results)

    return _init_fn(key, shapes_and_dtypes)


class _Init:
    @classmethod
    def abstract(cls, cfg: Config, *args, **kw):
        raise NotImplementedError

    @classmethod
    def shardings(cls, cfg: Config, *args, **kw):
        abstract = cls.abstract(cfg, *args, **kw)
        return jax.tree.map(
            lambda info: logical_to_sharding(info.logical_axes, cfg.mesh, cfg.rules),
            abstract, 
            is_leaf=is_param,
        )

    @classmethod
    def init(cls, key: random.PRNGKey, cfg: Config, *args, **kw):
        """Returns a pytree of randomly-initialized jax.Arrays corresponding to abstract()."""
        abstract = cls.abstract(cfg, *args, **kw)
        shardings = jax.tree.map(
            lambda info: logical_to_sharding(info.logical_axes, cfg.mesh, cfg.rules), abstract, is_leaf=is_param
        )
        abstract_leaves, abstract_struct = jax.tree.flatten(abstract, is_leaf=is_param)
        shardings_leaves = jax.tree.leaves(shardings, is_leaf=is_param)
        return jax.tree.unflatten(abstract_struct, _init_leaves(key, tuple(abstract_leaves), tuple(shardings_leaves)))


@partial(jax_pytree_struct, meta_fields=("out_scaling", "scale_expand_dims"))
class QuantArray:
    quant: jax.Array | ArrayInfo
    scale: jax.Array | ArrayInfo
    out_scaling: bool = False
    scale_expand_dims: int | tuple[int, ...] = ()
    shape = property(lambda self: self.quant.shape)
    ndim = property(lambda self: self.quant.ndim)


def einsum(subscripts: str, lhs: jax.Array, rhs: jax.Array | QuantArray, out_sharding: P | None = None):
    """jnp.einsum wrapper that handles regular arrays and QuantArrays"""
    if is_type(rhs, QuantArray):
        scale = jnp.expand_dims(rhs.scale, rhs.scale_expand_dims)
        if rhs.out_scaling:
            return jnp.einsum(subscripts, lhs, rhs.quant, out_sharding=out_sharding) * scale
        else:
            return jnp.einsum(subscripts, lhs * scale, rhs.quant, out_sharding=out_sharding)
    else:
        return jnp.einsum(subscripts, lhs, rhs, out_sharding=out_sharding)


_int8_quant_init = lambda key, shape, dtype=jnp.int8: random.randint(key, shape, -128, 128, dtype=dtype)
_int8_scale_init = lambda key, shape, dtype: random.normal(key, shape, dtype=dtype) / math.sqrt(math.prod(shape)) / 127


def quantize(x: jax.Array | ArrayInfo, axis: int | tuple[int, ...], scale_dtype=jnp.float16, zero_init: bool = False):
    if is_type(x, QuantArray):
        raise ValueError("Attempting to quantize an already quantized QuantArray.")
    if not isinstance(axis, (list, tuple)):
        axis = (axis,)
    axis = tuple(z % len(x.shape) for z in axis)

    if isinstance(x, jax.Array):
        axis = tuple(z % x.ndim for z in axis)
        amax = jnp.max(jnp.abs(x), axis=axis, keepdims=True)
        scale = (amax / 127.0 + jnp.finfo(scale_dtype).tiny).astype(scale_dtype)
        quant = jnp.round(x / scale).astype(jnp.int8)
        scale = scale.reshape([z for i, z in enumerate(scale.shape) if i not in axis])
        return quant, scale

    if is_type(x, ArrayInfo):
        new_shape = tuple(ax for i, ax in enumerate(x.shape) if i not in axis)
        new_logical_axes = tuple(ax for i, ax in enumerate(x.logical_axes) if i not in axis)
        if zero_init:
            quant_init, scale_init = jax.nn.initializers.zeros, jax.nn.initializers.ones
        else:
            quant_init, scale_init = _int8_quant_init, _int8_scale_init
        quant = dataclasses.replace(x, shape=x.shape, dtype=jnp.int8, initializer=quant_init)
        scale = ArrayInfo(new_shape, scale_dtype, new_logical_axes, scale_init)
        return quant, scale
    raise ValueError(f"quantize got unexpected type: {type(x)}")


def update_slice(x: jax.Array | QuantArray, y: jax.Array, pos: int, update_axis: int, quant_axis: int = -1):
    """dynamic_update_slice wrapper that handles regular arrays and QuantArrays"""
    if is_type(x, QuantArray):
        assert x.quant.ndim == y.ndim
        quant_axis, update_axis = quant_axis % x.quant.ndim, update_axis % x.quant.ndim
        y_quant, y_scale = quantize(y, axis=quant_axis, scale_dtype=x.scale.dtype)
        y_quant = reshard(y_quant.astype(x.quant.dtype), jax.typeof(x.quant).sharding.spec)
        y_scale = reshard(y_scale.astype(x.scale.dtype), jax.typeof(x.scale).sharding.spec)
        new_quant = jax.lax.dynamic_update_slice_in_dim(x.quant, y_quant, pos, axis=update_axis)
        scale_update_axis = [ax for ax in range(x.quant.ndim) if ax != quant_axis][update_axis]
        new_scale = jax.lax.dynamic_update_slice_in_dim(x.scale, y_scale, pos, axis=scale_update_axis)
        return dataclasses.replace(x, quant=new_quant, scale=new_scale)
    else:
        assert x.ndim == y.ndim
        y = reshard(y.astype(x.dtype), jax.typeof(x).sharding.spec)
        return jax.lax.dynamic_update_slice_in_dim(x, y, pos, axis=update_axis)


@partial(jax_pytree_struct, meta_fields=())
@dataclasses.dataclass
class MoeLayer(_Init):
    gate: jax.Array | ArrayInfo | QuantArray
    up: jax.Array | ArrayInfo | QuantArray
    down: jax.Array | ArrayInfo | QuantArray

    @classmethod
    def abstract(cls, cfg: Config) -> "MoeLayer":
        _init = lambda *out_axes: jax.nn.initializers.he_normal(in_axis=0, out_axis=out_axes)
        layer = MoeLayer(
            gate=ArrayInfo((cfg.embed, cfg.ffw_size), cfg.dtype, ("embed_up", "ffw_up"), _init(1)),
            up=ArrayInfo((cfg.embed, cfg.ffw_size), cfg.dtype, ("embed_up", "ffw_up"), _init(1)),
            down=ArrayInfo((cfg.ffw_size, cfg.embed), cfg.dtype, ("ffw_down", "embed_down"), _init(1)),
        )
        return layer


@partial(jax_pytree_struct, meta_fields=())
@dataclasses.dataclass
class Layer(_Init):
    q: jax.Array | ArrayInfo | QuantArray
    k: jax.Array | ArrayInfo | QuantArray
    v: jax.Array | ArrayInfo | QuantArray
    o: jax.Array | ArrayInfo | QuantArray
    moe_router: jax.Array | ArrayInfo | QuantArray
    moe_layers: list[MoeLayer]
    attn_pre_gamma: jax.Array | ArrayInfo
    attn_post_gamma: jax.Array | ArrayInfo

    @classmethod
    def abstract(cls, cfg: Config) -> "Layer":
        _init = lambda *out_axes: jax.nn.initializers.he_normal(in_axis=0, out_axis=out_axes)
        layer = Layer(
            q=ArrayInfo((cfg.embed, cfg.q_heads, cfg.head_dim), cfg.dtype, ("qkv_embed", "q_heads", "head_dim"), _init(1, 2)),
            k=ArrayInfo((cfg.embed, cfg.kv_heads, cfg.head_dim), cfg.dtype, ("qkv_embed", "kv_heads", "head_dim"), _init(1, 2)),
            v=ArrayInfo((cfg.embed, cfg.kv_heads, cfg.head_dim), cfg.dtype, ("qkv_embed", "kv_heads", "head_dim"), _init(1, 2)),
            o=ArrayInfo((cfg.q_heads, cfg.head_dim, cfg.embed), cfg.dtype, ("o_heads", "head_dim", "o_embed"), _init(1, 2)),
            moe_router=ArrayInfo((cfg.embed, cfg.moe_num_experts), cfg.dtype, ("embed_up", "experts"), _init(1)),
            moe_layers=[MoeLayer.abstract(cfg) for _ in range(cfg.moe_num_experts)],
            attn_pre_gamma=ArrayInfo((cfg.embed,), cfg.dtype, ("act_embed",), jax.nn.initializers.constant(1.0)),
            attn_post_gamma=ArrayInfo((cfg.embed,), cfg.dtype, ("act_embed",), jax.nn.initializers.constant(1.0)),
        )
        layer = cls.quantize(layer, cfg)
        return layer

    @staticmethod
    def quantize(layer: "Layer", cfg: Config):
        if not cfg.quant_layer:
            return layer
        scale_dtype = cfg.quant_scale_dtype
        quant_moe_layers = []
        for moe_layer in layer.moe_layers:
            quant_moe_layers.append(
                dataclasses.replace(
                    moe_layer,
                    gate=QuantArray(*quantize(moe_layer.gate, 0, scale_dtype), out_scaling=True),
                    up=QuantArray(*quantize(moe_layer.up, 0, scale_dtype), out_scaling=True),
                    down=QuantArray(*quantize(moe_layer.down, 0, scale_dtype), out_scaling=True),
                )
            )
        return dataclasses.replace(
            layer,
            q=QuantArray(*quantize(layer.q, (1, 2), scale_dtype)),
            k=QuantArray(*quantize(layer.k, (1, 2), scale_dtype)),
            v=QuantArray(*quantize(layer.v, (1, 2), scale_dtype)),
            o=QuantArray(*quantize(layer.o, (0, 1), scale_dtype), out_scaling=True),
            moe_router=QuantArray(*quantize(layer.moe_router, 0, scale_dtype), out_scaling=True),
            moe_layers=quant_moe_layers,
        )


@partial(jax_pytree_struct, meta_fields=())
@dataclasses.dataclass
class Weights(_Init):
    layers: list[Layer]
    embedding: jax.Array | ArrayInfo
    gamma_final: jax.Array | ArrayInfo
    lm_head: jax.Array | ArrayInfo

    @classmethod
    def abstract(cls, cfg: Config):
        layers = [Layer.abstract(cfg) for _ in range(cfg.num_layers)]
        init = lambda in_axis, out_axis: jax.nn.initializers.he_normal(in_axis=in_axis, out_axis=out_axis)
        return Weights(
            layers=layers,
            embedding=ArrayInfo((cfg.vocab_size, cfg.embed), cfg.dtype, ("vocab_in", "vocab_in"), init(0, 1)),
            gamma_final=ArrayInfo((cfg.embed,), cfg.dtype, ("act_embed",), jax.nn.initializers.constant(1.0)),
            lm_head=ArrayInfo((cfg.embed, cfg.vocab_size), cfg.dtype, ("vocab_in", "vocab_out"), init(1, 0)),
        )


@partial(jax_pytree_struct, meta_fields=())
@dataclasses.dataclass
class KVCache(_Init):
    k: list[jax.Array]
    v: list[jax.Array]
    length: jax.Array
    starts: jax.Array

    @classmethod
    def abstract(cls, cfg: Config, batch_size: int, max_seq_len: int):
        val_info = ArrayInfo(
            (batch_size, cfg.kv_heads, max_seq_len, cfg.head_dim),
            cfg.dtype,
            ("batch", "kv_heads", "sequence", "head_dim"),
            jax.nn.initializers.zeros,
        )
        cache = KVCache(
            k=[val_info for _ in range(cfg.num_layers)],
            v=[val_info for _ in range(cfg.num_layers)],
            length=ArrayInfo((), jnp.int32, (), jax.nn.initializers.zeros),
            starts=ArrayInfo((batch_size,), jnp.int32, ("batch",), jax.nn.initializers.zeros),
        )
        if cfg.quant_cache:
            _quantize = partial(quantize, axis=-1, scale_dtype=cfg.quant_scale_dtype, zero_init=True)
            cache = dataclasses.replace(
                cache,
                k=[
                    QuantArray(*_quantize(cache.k[idx]), out_scaling=True, scale_expand_dims=(-2, -3))
                    for idx in range(len(cache.k))
                ],
                v=[
                    QuantArray(*_quantize(cache.v[idx]), out_scaling=False, scale_expand_dims=(-2, -3))
                    for idx in range(len(cache.v))
                ],
            )
        return cache

    @property
    def time_axis(self) -> int:
        return 2


def segment_ids_to_positions(segment_ids):
    def scan_fun(a, b):
        return ((a[0] + 1) * (a[1] == b[1]) + b[0], b[1])

    vals = (jnp.zeros_like(segment_ids), segment_ids)
    return jnp.array(jax.lax.associative_scan(scan_fun, vals, axis=-1)[0], dtype="int32")


def _generate_pos_embeddings(positions: jax.Array, features: int, cfg: Config) -> tuple[jax.Array, jax.Array]:
    fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
    timescale = cfg.rope_theta ** fraction
    rotational_frequency = 1.0 / timescale
    sinusoid_inp = jnp.einsum(
        "BT,k->BTk",
        positions,
        rotational_frequency,
        precision=jax.lax.Precision.HIGHEST,
        out_sharding=P(None, None, None),
    )
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def apply_rotary_embedding(x, sin, cos):
    assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
    x1, x2 = jnp.split(x, 2, axis=-1)
    sin, cos = sin[:, None, :, :], cos[:, None, :, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

def make_attention_mask(q_len, k_len, q_segment_ids, k_segment_ids, q_offset, causal: bool, sliding_window_size: int):
    segment_mask = q_segment_ids[:, :, None] == k_segment_ids[:, None, :]
    segment_mask = segment_mask[:, None, :, :]

    if causal:
        qk = (1, 1, q_len, k_len)
        q_iota = jax.lax.broadcasted_iota(jnp.int32, qk, 2)
        k_iota = jax.lax.broadcasted_iota(jnp.int32, qk, 3)
        q_positions = q_iota + q_offset[:, None, None, None]
        causal_mask = q_positions >= k_iota
        if sliding_window_size > 0:
            causal_mask = jnp.logical_and(causal_mask, (q_positions - k_iota) < sliding_window_size)
        combined_mask = jnp.logical_and(segment_mask, causal_mask)
        return combined_mask
    else:
        return segment_mask


def _attention(
    q: jax.Array,
    k: jax.Array | tuple[jax.Array, jax.Array],
    v: jax.Array | tuple[jax.Array, jax.Array],
    q_segment_ids: jax.Array,
    k_segment_ids: jax.Array,
    q_offset: jax.Array,
    cfg: Config,
) -> jax.Array:
    scale = cfg.head_dim ** -0.5
    b, qh, t, d = q.shape
    _, kh, T, _ = k.shape
    q_ = q.reshape((b, kh, qh // kh, t, d))
    qk = einsum("bhgtd,bhTd->bhgtT", q_, k) * scale
    qk = qk.reshape((b, qh, t, T))
    mask = make_attention_mask(t, T, q_segment_ids, k_segment_ids, q_offset, cfg.causal, cfg.sliding_window_size)
    qk = jnp.where(mask, qk, -1e30)
    attn = jax.nn.softmax(qk.astype(jnp.float32), axis=-1)
    attn_ = attn.reshape((b, kh, qh // kh, t, T))
    qkv = einsum("bhgtT,bhTd->bhgtd", attn_, v).astype(cfg.dtype)
    return qkv.reshape((b, qh, t, d))


attention = auto_axes(_attention, out_sharding=P(BATCH_AXIS_NAME, TENSOR_AXIS_NAME, None, None))

def attention_kernel(q, k, v, q_segment_ids, kv_segment_ids, q_offset, starts, lengths, cfg: Config):
    k, k_scale = (k.quant, k.scale) if is_type(k, QuantArray) else (k, None)
    v, v_scale = (v.quant, v.scale) if is_type(v, QuantArray) else (v, None)
    assert q.shape[-3] % k.shape[-3] == 0
    scale = q.shape[-1] ** -0.5
    l2p = lambda *logical: logical_to_physical(logical, cfg.rules)
    kv_repeats = q.shape[-3] // k.shape[-3]
    q_spec = P(*(l2p("batch", "kv_heads") + tuple(set(*l2p("q_heads")) - set(*l2p("kv_heads"))) + l2p("sequence", "head_dim")))
    q_shape__ = q.shape
    q = jax.lax.reshape(q, (q.shape[:-3] + (k.shape[-3], kv_repeats, q.shape[-2], q.shape[-1])), out_sharding=q_spec)
    in_specs = (
        q_spec,
        l2p("batch", "kv_heads", "sequence", "head_dim"),
        l2p("batch", "kv_heads", "sequence", "head_dim"),
        l2p("batch", "sequence"),
        l2p("batch", "sequence"),
        l2p("batch"),
        l2p("batch"),
    )
    in_specs += (None if k_scale is None else l2p("batch", "kv_heads", "sequence"),)
    in_specs += (None if v_scale is None else l2p("batch", "kv_heads", "sequence"),)
    out_specs = q_spec

    @partial(shard_map, mesh=cfg.mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)
    def _f(q, k, v, q_segment_ids, kv_segment_ids, starts, lengths, k_scale, v_scale):
        q_org_shape = q.shape
        if q.shape[-2] != 1:
            mask = mask_lib.MultiHeadMask([mask_lib.CausalMask((q.shape[-2], k.shape[-2])) for _ in range(q.shape[-3])])
            block_q, block_kv = min(q.shape[-2], 512), min(k.shape[-2], 1024)
            block_sizes = splash.BlockSizes(block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv)
            attn_fn = splash.make_splash_mqa_single_device(mask=mask, block_sizes=block_sizes)
            attn_fn = jax.vmap(jax.vmap(attn_fn, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, 0))
            segment_ids = splash.SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
            if k_scale is not None:
                k = (k * k_scale[..., None]).astype(jnp.bfloat16)
            if v_scale is not None:
                v = (v * v_scale[..., None]).astype(jnp.bfloat16)
            ret = attn_fn(q * scale, k, v, segment_ids)
        else:
            assert q.shape[-2] == 1, "This is a decode kernel, q.shape[-2] must be 1"
            q = q[..., 0, :]
            in_axes = (1, 1, 1, None, None)
            in_axes += ((None if k_scale is None else 1),)
            in_axes += ((None if v_scale is None else 1),)
            hyperparams = dict(scale=scale, block_kv=512, block_bs=32)
            ret = jax.vmap(partial(ragged_attention.ragged_decode_fwd, **hyperparams), in_axes=in_axes, out_axes=1)(
                q, k, v, starts, lengths, k_scale, v_scale
            )
        return ret.reshape(q_org_shape)

    lengths = jnp.broadcast_to(lengths, starts.shape)
    ret = _f(q, k, v, q_segment_ids, kv_segment_ids, starts, lengths, k_scale, v_scale).astype(jnp.bfloat16)
    return jax.lax.reshape(ret, q_shape__, out_sharding=l2p("batch", "q_heads", "sequence", "head_dim"))


def rms_norm(x: jax.Array, gamma: jax.Array) -> jax.Array:
    rms = jnp.sqrt(jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6)
    return jnp.astype(gamma * x / rms, jnp.bfloat16)


def attention_block(
    x: jax.Array,
    segment_ids: jax.Array,
    layer: Layer,
    sin: jax.Array,
    cos: jax.Array,
    cfg: Config,
    cache: KVCache | None = None,
    idx: int | None = None,
):
    x = x.astype(cfg.dtype)
    with jax.named_scope("qkv_matmul"):
        q = einsum("btd,dhq->bhtq", x, layer.q).astype(cfg.dtype)
        k = einsum("btd,dhq->bhtq", x, layer.k).astype(cfg.dtype)
        v = einsum("btd,dhq->bhtq", x, layer.v).astype(cfg.dtype)
    with jax.named_scope("rope"):
        q, k = apply_rotary_embedding(q, sin, cos), apply_rotary_embedding(k, sin, cos)
    with jax.named_scope("cache_update"):
        if cache is not None:
            k = update_slice(cache.k[idx], k, cache.length, update_axis=cache.time_axis, quant_axis=-1)
            v = update_slice(cache.v[idx], v, cache.length, update_axis=cache.time_axis, quant_axis=-1)
            time_indices = jnp.arange(0, v.shape[cache.time_axis])[None, :]
            q_segment_ids = jnp.where(segment_ids != 0, 1, 0)
            incremental_position = jnp.max(_length_minus_padding(segment_ids))
            k_segment_ids = ((time_indices >= cache.starts[:, None]) & (time_indices < (cache.length + incremental_position))).astype(jnp.int32)
            q_offset = cache.length[None]
            starts, lengths = cache.starts, (cache.length + incremental_position)[None]
        else:
            q_segment_ids, k_segment_ids = segment_ids, segment_ids
            q_offset = jnp.zeros(x.shape[0], dtype=jnp.int32)
            starts, lengths = _count_left_padding(k_segment_ids, 0), _length_minus_padding(k_segment_ids)
    with jax.named_scope("attention"):
        if (cfg.use_prefill_attn_kernel and q.shape[-2] != 1) or (cfg.use_decode_attn_kernel and q.shape[-2] == 1):
            attn_out = attention_kernel(q, k, v, q_segment_ids, k_segment_ids, q_offset, starts=starts, lengths=lengths, cfg=cfg)
        else:
            attn_out = attention(q, k, v, q_segment_ids, k_segment_ids, q_offset, cfg)
    with jax.named_scope("projection"):
        attn_out = einsum("bhtq,hqd->btd", attn_out, layer.o).astype(cfg.dtype)
    return attn_out, k, v


def ffn_block(x: jax.Array, layer: Layer, cfg: Config):
    router_logits = einsum("btd,de->bte", x, layer.moe_router)
    routing_weights, selected_experts = jax.lax.top_k(router_logits, k=cfg.moe_num_experts_per_tok)
    routing_weights = jax.nn.softmax(routing_weights, axis=-1)
    
    final_hidden_states = jnp.zeros_like(x)
    for i in range(cfg.moe_num_experts):
        expert_layer = layer.moe_layers[i]
        expert_mask = jnp.any(selected_experts == i, axis=-1)
        
        # Always compute (masking will zero out inactive experts)
        ff_gate = jax.nn.silu(einsum("btd,df->btf", x, expert_layer.gate)).astype(cfg.dtype)
        ff_up = einsum("btd,df->btf", x, expert_layer.up).astype(cfg.dtype)
        ff_out = einsum("btf,fd->btd", ff_gate * ff_up, expert_layer.down).astype(cfg.dtype)
        
        # Create expert selection mask and get routing weights
        expert_selection = (selected_experts == i)  # [batch, seq, num_active]
        expert_routing_weights = jnp.sum(routing_weights * expert_selection, axis=-1)  # [batch, seq]
        
        # Apply mask and routing weights
        expert_weight = expert_routing_weights[..., None] * expert_mask[..., None]  # [batch, seq, 1]
        final_hidden_states += ff_out * expert_weight

    return final_hidden_states


def forward_layer(
    x: jax.Array,
    segment_ids: jax.Array,
    layer: Layer,
    sin: jax.Array,
    cos: jax.Array,
    idx: int,
    cfg: Config,
    cache: KVCache | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x = x.astype(cfg.dtype)
    with jax.named_scope("attn_pre_norm"):
        attn_in = rms_norm(x, layer.attn_pre_gamma)
    attn_out, k, v = attention_block(attn_in, segment_ids, layer, sin, cos, cfg, cache, idx)
    with jax.named_scope("residual"):
        x = x + attn_out.astype(cfg.dtype)
    with jax.named_scope("attn_post_norm"):
        ff_in = rms_norm(x, layer.attn_post_gamma)
    with jax.named_scope("ffn"):
        ff_out = ffn_block(ff_in, layer, cfg)
    with jax.named_scope("residual"):
        x = x + ff_out.astype(cfg.dtype)
    return x, k, v


def forward(
    x: jax.Array,
    segment_ids: jax.Array,
    weights: Weights,
    cfg: Config,
    cache: KVCache | None = None,
):
    l2p = lambda *args: logical_to_physical(args, cfg.rules)
    x = weights.embedding.at[x, :].get(out_sharding=l2p("batch", "sequence", "act_embed"))
    batch = x.shape[0]
    positions = segment_ids_to_positions(segment_ids)
    if cache is not None:
        start_indices = jnp.where(cache.length != 0, cache.length - cache.starts, 0)
    else:
        start_indices = jnp.zeros((batch,), dtype=jnp.int32)
    positions = start_indices[:, None] + positions
    sin, cos = _generate_pos_embeddings(positions, cfg.head_dim, cfg)
    sin, cos = sin.astype(cfg.dtype), cos.astype(cfg.dtype)
    for idx, layer in enumerate(weights.layers):
        x, k, v = forward_layer(x, segment_ids, layer, sin, cos, idx, cfg, cache)
        if cache is not None:
            cache.k[idx], cache.v[idx] = k, v
    x = rms_norm(x, weights.gamma_final)
    logits = einsum("btd,dv->btv", x, weights.lm_head)
    if cache is not None:
        cache = dataclasses.replace(cache, length=cache.length + jnp.max(_length_minus_padding(segment_ids)))
        return logits, cache
    return logits

def save_pytree(data, path):
    import orbax.checkpoint as ocp

    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(epath.Path(path), data, ocp.args.PyTreeSave(data, ocdbt_target_data_file_size=1024 * 1024 * 100))

def load_pytree(path, sharding=None):
    import orbax.checkpoint as ocp

    item, transforms = sharding, None
    restore_args = jax.tree.map(lambda s: ocp.ArrayRestoreArgs(sharding=s), sharding)
    with ocp.PyTreeCheckpointer() as ckptr:
        return ckptr.restore(
            epath.Path(path), ocp.args.PyTreeRestore(item=item, transforms=transforms, restore_args=restore_args)
        )


@partial(jax.jit, static_argnums=(1, 2))
def prepare_chunk(chunk, pad_to: int, pad_id: int):
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    chunk = jnp.pad(chunk, [(0, 0), (0, pad_to - chunk.shape[-1])])
    segment_ids = jnp.where(chunk != pad_id, 1, 0).astype(jnp.int32)
    return chunk, segment_ids


def prefill(tokens: jax.Array, weights: Weights, cache: KVCache, cfg: Config, pad_id: int = 0):
    assert tokens.shape[-1] <= cfg.max_seq_len
    pad_to = 2 ** math.ceil(math.log2((tokens.shape[-1])))
    with use_mesh(cfg.mesh):
        prompt, prompt_segment_ids = prepare_chunk(tokens, pad_to=pad_to, pad_id=pad_id)
        assert prompt.ndim == 2
        cache_shardings = KVCache.shardings(cfg, cache.k[0].shape[0], cache.k[0].shape[cache.time_axis])
        logits_shardings = jax.sharding.NamedSharding(cfg.mesh, P(BATCH_AXIS_NAME, None, TENSOR_AXIS_NAME))
        cache = dataclasses.replace(
            cache, length=jnp.zeros_like(cache.length), starts=_count_left_padding(tokens, pad_id=pad_id)
        )
        logits, cache = jax.jit(forward, donate_argnums=(4,), out_shardings=(logits_shardings, cache_shardings))(
            prompt, prompt_segment_ids, weights, cfg, cache
        )
        next_tokens = jax.jit(jnp.argmax, static_argnames=("axis",))(logits, axis=-1)
        return next_tokens, logits, cache


@partial(jax.jit, donate_argnames=("cache",))
def decode_step(last_tokens: jax.Array, weights: Weights, cache: KVCache, cfg: Config):
    assert last_tokens.ndim == 2
    segment_ids = jnp.ones(last_tokens.shape, dtype=jnp.int32)
    next_logits, cache = forward(last_tokens, segment_ids, weights, cfg, cache)
    next_tokens = jnp.argmax(next_logits, -1)
    next_tokens = reshard(next_tokens, P())
    return next_tokens, cache


def generate(
    prompt: jax.Array,
    weights: Weights,
    cache: KVCache,
    cfg: Config,
    max_new_tokens: int,
    pad_id: int = 0,
    eos_id: int = 2,
):
    next_tokens, _, cache = prefill(prompt, weights, cache, cfg, pad_id)
    next_tokens = next_tokens[:, -1:]
    generated_tokens = [next_tokens]
    for _ in range(max_new_tokens - 1):
        next_tokens, cache = decode_step(next_tokens, weights, cache, cfg)
        generated_tokens.append(next_tokens)
        if jnp.all(next_tokens == eos_id):
            break
    return jnp.concatenate(generated_tokens, axis=-1)
