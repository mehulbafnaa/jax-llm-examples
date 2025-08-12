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

import math
import time
from functools import partial

import jax
from jax import numpy as jnp
from jax import random
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P, NamedSharding

NUM_LANES = 128


@partial(jax.named_call, name="ragged_decode_kernel")
def ragged_decode_kernel_fwd(
    # prefetch scalars:
    start_ref,  # [bs]
    length_ref,  # [bs]
    chunked_start_ref,  # [bs // block_bs]
    chunked_length_ref,  # [bs // block_bs]
    # inputs:
    q_ref,  # [bs // block_bs, heads, head_dim]
    k_ref,  # [bs // block_bs, block_kv, head_dim]
    v_ref,  # [bs // block_bs, block_kv, head_dim]
    k_scale_ref,  # optional [bs // block_bs, heads] not None if k is quantized
    v_scale_ref,  # optional [bs // block_bs, heads] not None if v is quantized
    qk_prev_ref,  # optional [bs // block_vs, heads, block_kv] not None if some qk is precomputed (Deepseek on TPU)
    # outputs:
    o_ref,  # [bs // block_bs, heads, head_dim]
    # scratch memory:
    o_scratch_ref,  # [bs // block_bs, heads, head_dim]
    l_scratch_ref,  # [bs // block_bs, heads, TPU_MIN_SIZE]
    m_scratch_ref,  # [bs // block_bs, heads, TPU_MIN_SIZE]
    # parameters:
    kv_seq_len: int,
    block_kv: int,
    block_bs: int,
    scale: float,
    scale_qk_not_k: bool = True,
    scale_s_not_v: bool = True,
):
    del chunked_start_ref, chunked_length_ref
    mask_value = jnp.finfo(o_scratch_ref.dtype).min
    b_, i = pl.program_id(0), pl.program_id(1)

    @pl.when(i == 0)
    def init():
        m_scratch_ref[...] = jnp.full_like(m_scratch_ref, -jnp.inf)
        l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
        o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)

    def resize(x, new_size_in_dim, axis=-1):
        """Resize the shape of array x to the target size along axis `axis`."""
        if x.shape[axis] > new_size_in_dim:
            assert axis in (-1, x.ndim - 1)
            return x[..., :new_size_in_dim]
        return pltpu.repeat(x, new_size_in_dim // x.shape[axis], axis=axis % x.ndim)

    def loop_fn(b, _):
        b_global = block_bs * b_ + b
        start, length = start_ref[b_global], length_ref[b_global]
        block_start, block_end = i * block_kv, (i + 1) * block_kv
        should_compute = (start < length) & ((block_start < length) & (block_end >= start))

        @pl.when(should_compute)
        def compute():
            # compute qk
            q, k = q_ref[b, ...], k_ref[b, ...]
            if k_scale_ref is not None and not scale_qk_not_k:
                k = k * k_scale_ref[b, ...].astype(jnp.float32).reshape(k.shape[:-1] + (1,)).astype(jnp.bfloat16)
            qk = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            if k_scale_ref is not None and scale_qk_not_k:
                qk = qk * k_scale_ref[b, ...]
            if qk_prev_ref is not None:
                qk += qk_prev_ref[b, ...]
            qk *= scale
            indices = i * block_kv + jax.lax.broadcasted_iota(jnp.int32, qk.shape, dimension=1)
            mask = (indices >= start) & (indices < length)
            qk += jnp.where(mask, 0, mask_value)

            # adjust maximum shift value, shift and softmax
            m_prev, l_prev = m_scratch_ref[b, ...], l_scratch_ref[b, ...]
            m_curr = resize(jnp.max(qk, axis=-1)[:, None], m_prev.shape[-1])
            m_next = jnp.maximum(m_prev, m_curr)
            s_curr = jnp.exp(qk - resize(m_next, qk.shape[-1]))
            l_curr = jax.lax.broadcast_in_dim(jnp.sum(s_curr, axis=-1), l_prev.shape, (0,))

            # compute the (qk v)
            v = v_ref[b, ...]
            if v_scale_ref is not None and not scale_s_not_v:
                v = v * v_scale_ref[b, ...].astype(jnp.float32).reshape(v.shape[:-1] + (1,)).astype(jnp.bfloat16)
            elif v_scale_ref is not None and scale_s_not_v:
                s_curr = s_curr * v_scale_ref[b, ...]
            o_curr = jax.lax.dot_general(s_curr, v, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)

            # accumulate the results
            o_prev = o_scratch_ref[b, ...]
            m_next = jnp.maximum(m_prev, m_curr)
            alpha = jnp.exp(m_prev - m_next)
            l_next = l_prev * alpha + l_curr
            l_next_safe = l_next
            o_next = resize(alpha, o_prev.shape[-1]) * o_prev + o_curr

            # store scratch values
            m_scratch_ref[b, ...] = m_next
            l_scratch_ref[b, ...] = l_next_safe
            o_scratch_ref[b, ...] = o_next

    jax.lax.fori_loop(0, block_bs, loop_fn, init_val=None)

    @pl.when(i == (kv_seq_len // block_kv) - 1)
    def done():
        l = l_scratch_ref[...]
        l_inv = jnp.where(l == 0.0, 1.0, 1.0 / l)
        o_ref[...] = (o_scratch_ref[...] * resize(l_inv, o_scratch_ref.shape[-1])).astype(o_ref.dtype)


def ragged_decode_fwd(
    q: jax.Array,  # [bs, q_heads, head_dim]
    k: jax.Array,  # [bs, kv_seq_len, head_dim]
    v: jax.Array,  # [bs, kv_seq_len, head_dim]
    starts: jax.Array | None = None,  # [bs]
    lengths: jax.Array | None = None,  # [bs]
    k_scale: jax.Array | None = None,  # [bs, kv_seq_len]
    v_scale: jax.Array | None = None,  # [bs, kv_seq_len]
    qk_prev: jax.Array | None = None,  # [bs, q_heads, kv_seq_len]
    block_bs: int = 4,
    block_kv: int = 256,
    scale: float | None = None,
    scale_qk_not_k: bool = True,
    scale_s_not_v: bool = True,
    interpret: bool = False,
):
    scale = math.sqrt(q.shape[-1]) if scale is None else scale
    bs_q, q_heads, head_dim_q = q.shape
    bs_k, kv_seq_len_k, head_dim_k = k.shape
    assert bs_q == bs_k and head_dim_q == head_dim_k
    bs, kv_seq_len = bs_q, kv_seq_len_k
    bs_v, kv_seq_len_v, head_dim_v = v.shape
    assert bs == bs_v and kv_seq_len == kv_seq_len_v

    block_bs = min(bs, block_bs)
    assert bs % block_bs == 0

    if starts is None:
        starts = jnp.zeros((bs,), dtype=jnp.int32)
    if lengths is None:
        lengths = kv_seq_len * jnp.ones((bs,), dtype=jnp.int32)

    assert starts.ndim == 1 and starts.size == bs
    assert lengths.ndim == 1 and lengths.size == bs
    block_kv = min(kv_seq_len, block_kv)
    assert kv_seq_len % block_kv == 0

    chunked_starts = jnp.min(starts.reshape((-1, block_bs)), axis=-1)
    chunked_lengths = jnp.max(lengths.reshape((-1, block_bs)), axis=-1)

    def kv_prefetch_map(b, i, starts_ref, lengths_ref, chunked_starts_ref, chunked_lengths_ref):
        del starts_ref, lengths_ref
        start, length = chunked_starts_ref[b], chunked_lengths_ref[b]
        s_idx = i * block_kv
        last_batch, seq_done = b == (bs // block_bs) - 1, s_idx > length
        start_next = chunked_starts_ref[b + (~last_batch)]
        first_start_i, next_start_i = start // block_kv, start_next // block_kv
        b = jnp.where(seq_done & (~last_batch), b + 1, b)
        i = jnp.where(seq_done, jnp.where(last_batch, i, next_start_i), jnp.maximum(first_start_i, i))
        i = jnp.where(last_batch & seq_done, pl.cdiv(length, block_kv) - 1, i)
        return b, i, 0

    def kv_scale_prefetch_map(b, i, starts_ref, lengths_ref, chunked_starts_ref, chunked_lengths_ref):
        b_, i_, _ = kv_prefetch_map(b, i, starts_ref, lengths_ref, chunked_starts_ref, chunked_lengths_ref)
        return b_, 0, i_

    in_specs = []
    in_specs += [pl.BlockSpec((block_bs, q_heads, q.shape[-1]), lambda b, i, *_: (b, 0, 0))]  # q
    in_specs += [pl.BlockSpec((block_bs, block_kv, k.shape[-1]), kv_prefetch_map)]  # k
    in_specs += [pl.BlockSpec((block_bs, block_kv, head_dim_v), kv_prefetch_map)]  # v
    if k_scale is not None:
        in_specs += [pl.BlockSpec((block_bs, 1, block_kv), kv_scale_prefetch_map)]
        k_scale = k_scale.reshape(k_scale.shape[:-1] + (1, k_scale.shape[-1])).astype(jnp.bfloat16)
    else:
        in_specs += [None]

    if v_scale is not None:
        in_specs += [pl.BlockSpec((block_bs, 1, block_kv), kv_scale_prefetch_map)]
        v_scale = v_scale.reshape(v_scale.shape[:-1] + (1, v_scale.shape[-1])).astype(jnp.bfloat16)
    else:
        in_specs += [None]

    if qk_prev is not None:
        qk_prev_prefetch_map = kv_scale_prefetch_map
        in_specs += [pl.BlockSpec((block_bs, q_heads, block_kv), qk_prev_prefetch_map)]
    else:
        in_specs += [None]

    out_shape = jax.ShapeDtypeStruct((bs, q_heads, head_dim_v), q.dtype)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=4,
        grid=(bs // block_bs, kv_seq_len // block_kv),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((block_bs, q_heads, head_dim_v), lambda b, i, *_: (b, 0, 0)),
        scratch_shapes=[
            pltpu.VMEM((block_bs, q_heads, head_dim_v), dtype=jnp.float32),
            pltpu.VMEM((block_bs, q_heads, NUM_LANES), dtype=jnp.float32),
            pltpu.VMEM((block_bs, q_heads, NUM_LANES), dtype=jnp.float32),
        ],
    )
    kernel = partial(
        ragged_decode_kernel_fwd,
        kv_seq_len=kv_seq_len,
        block_kv=block_kv,
        block_bs=block_bs,
        scale=scale,
        scale_qk_not_k=scale_qk_not_k,
        scale_s_not_v=scale_s_not_v,
    )
    attn = pl.pallas_call(kernel, grid_spec=grid_spec, out_shape=out_shape, interpret=interpret)(
        starts, lengths, chunked_starts, chunked_lengths, q, k, v, k_scale, v_scale, qk_prev
    )
    return attn
