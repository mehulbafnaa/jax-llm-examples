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

"""Minimal serving example for Gemma-4 (E2B/E4B/...).

Gemma-4 interleaves *local* (sliding-window) and *global* attention layers whose KV cache buffers have different
sequence lengths (and head dims). The other serving examples stack every layer's KV into a single array for faster
dispatch; that is impossible here, so this example keeps the per-layer buffers **unrolled** and uses the
`gemma_kvcache_{get,insert}_sequence` helpers in `serving_jax.attention_cache_utils`. Continuous batching with mixed
local/global attention is handled principally -- each layer inserts using its own window/cursor/starts.
"""

import dataclasses
import socket
import threading
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import serving_jax as serving
from gemma4_jax import model as g4jax
from jax import random
from jax.sharding import AxisType, set_mesh
from serving_jax import attention_cache_utils as attn_utils

Config = Any

jax.config.update("jax_explain_cache_misses", True)
# jax.config.update("jax_compilation_cache_dir", str(Path("~/.cache/jax").expanduser()))

try:  # newer JAX only
    my_id = int(socket.gethostname().split("-")[-1])
    my_ip = socket.getaddrinfo(socket.gethostname(), 80)[0][-1][0]
    jax.config.update("jax_cross_host_transfer_socket_address", f"{my_ip}:{17007 + my_id}")
    jax.config.update("jax_cross_host_transport_addresses", ",".join([f"{my_ip}:0"] * 8))
except:  # noqa: E722
    pass

EOS_TOKENS = (1, 106, 50)  # <eos>, <end_of_turn>, and the model's extra end token (see generation_config.json)


def encode_input(tokenizer, texts, pad_id: int = g4jax.PAD_ID):
    assert isinstance(texts, list)
    inputs = [
        tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True) for text in texts
    ]
    inputs = [getattr(text, "input_ids", text) for text in inputs]
    max_len = max([len(x) for x in inputs])
    return np.array([(max_len - len(x)) * [pad_id] + x for x in inputs])


tokenizer_encode = lambda tokenizer, text: encode_input(tokenizer, [text])[0].tolist()
tokenizer_decode = lambda tokenizer, tokens: tokenizer.decode(tokens)


def distributed_init(is_coordinator: bool):
    try:
        # for TPU / multi-host
        jax.distributed.initialize()

        # for GPU/CPU multi-host
        # process_idx = int(socket.gethostname().split("-")[-1]) - 1  # a scheme where hosts are (host-1, host-2, ...)
        # jax.distributed.initialize(os.environ["COORDINATOR_ADDRESS"], 2, process_idx)
    except Exception as e:  # noqa: E722 -- single-process (single node) runs don't need the distributed runtime
        print(f"---> Skipping jax.distributed.initialize (running single-process?): {e}")

    if not serving.SyncServer.broadcast("welcome", 0, is_coordinator, is_coordinator):
        raise ValueError("Neither this proccess nor any other processe is the main server, exactly one must.")


def load_tokenizer(ckpt_path: Path):
    return g4jax.load_tokenizer(
        ckpt_path / "tokenizer.json", ckpt_path / "tokenizer_config.json", ckpt_path / "chat_template.jinja"
    )


def build_config_and_weights(ckpt_path: Path, mesh, max_seq_len: int, use_prefill_attn_kernel: bool):
    """Load the config + weights for a (quant or unquant) Gemma-4 checkpoint onto `mesh`."""
    is_quant = "quant" in ckpt_path.name  # quantized checkpoints have the '-quant' suffix
    cfg = g4jax.load_config(ckpt_path / "config.json")
    cfg = dataclasses.replace(
        cfg,
        mesh=mesh,
        max_seq_len=max_seq_len,
        quant_moe=is_quant,  # E4B is dense (no MoE), but this keeps larger MoE variants working too
        quant_mlp=is_quant,
        quant_attn=False,
        quant_cache=False,  # keep the KV cache in bf16 so the unrolled buffers stay plain arrays
        use_prefill_attn_kernel=use_prefill_attn_kernel,
        use_decode_attn_kernel=False,
    )
    weights_formats, _ = g4jax.optimal_formats(cfg)
    weights = g4jax.load_pytree(ckpt_path, weights_formats)
    return cfg, weights, is_quant


def build_serving_loop(ckpt_path: Path, *, is_server: bool, max_seq_len: int, decode_batch_size: int,
                       prefill_batch_size: int, decode_steps: int, max_decode_length: int,
                       use_prefill_attn_kernel: bool):
    devices = jax.devices()  # this helps catch distributed errors quickly

    # A single shared mesh for both prefill and decode -- no cross-host transfer on a single (multi-GPU) node.
    mesh = jax.make_mesh((1, 1, len(devices)), ("x", "y", "z"), devices=devices, axis_types=(AxisType.Explicit,) * 3)

    cfg, weights, is_quant = build_config_and_weights(ckpt_path, mesh, max_seq_len, use_prefill_attn_kernel)
    print(f"---> Model config loaded ({'quantized' if is_quant else 'unquantized'} weights)")
    print(f"     {cfg.num_layers=} attention_types={''.join('G' if t=='global_attention' else 'L' for t in cfg.attention_types)}")
    print("---> Weights loaded")

    serve_cfg = serving.ServingConfig(
        decode_steps=decode_steps,
        decode_batch_size=decode_batch_size,
        prefill_batch_size=prefill_batch_size,
        max_decode_length=max_decode_length,
        eos_tokens=EOS_TOKENS,
        token_pad_idx=g4jax.PAD_ID,
        use_prefix_cache=False,  # disabled: mixed local/global cache sizes make single-axis prefix chunking ill-defined
        time_axis=2,
    )

    # Unrolled (per-layer) attention cache -- the gemma-specific get/insert keep the buffers un-stacked.
    with set_mesh(mesh):
        decode_cache = g4jax.KVCache.init(random.key(0), cfg, serve_cfg.decode_batch_size)
        prefill_cache = g4jax.KVCache.init(random.key(0), cfg, serve_cfg.prefill_batch_size)
    decode_cache = serving.AttentionWrapper(
        decode_cache, attn_utils.gemma_kvcache_get_sequence, attn_utils.gemma_kvcache_insert_sequences
    )
    prefill_cache = serving.AttentionWrapper(
        prefill_cache, attn_utils.gemma_kvcache_get_sequence, attn_utils.gemma_kvcache_insert_sequences
    )

    sampler = partial(jnp.argmax, axis=-1)

    @partial(jax.jit, donate_argnames=("cache",))
    def forward_fn(inputs, weights, cache, cfg):
        logits, cache = g4jax.forward(inputs, (inputs != g4jax.PAD_ID).astype(jnp.int32), weights, cfg, cache)
        return sampler(logits), cache

    # prefill and decode share the same mesh and weights (single-node, interleaved serving)
    serve_loop = serving.ServingLoop(
        serve_cfg, cfg, forward_fn, weights, prefill_cache, weights, decode_cache, is_server
    )
    print("---> Created the serving loop")
    return serve_loop, cfg


def main():
    parser = ArgumentParser()
    parser.add_argument("--server", action="store_true", help="Make this node the main server.", default=False)
    parser.add_argument("--ckpt-path", default="/workspace/gemma4_ckpts/jax/gemma4-jax-E4B-quant")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--prefill-batch-size", type=int, default=2)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--max-decode-length", type=int, default=256)
    parser.add_argument("--no-prefill-kernel", action="store_true", default=False)
    ARGS = parser.parse_args()

    distributed_init(ARGS.server)
    ckpt_path = Path(ARGS.ckpt_path).expanduser()
    assert ckpt_path.is_dir(), f"Checkpoint not found: {ckpt_path}"
    tokenizer = load_tokenizer(ckpt_path)

    serve_loop, _ = build_serving_loop(
        ckpt_path,
        is_server=ARGS.server,
        max_seq_len=ARGS.max_seq_len,
        decode_batch_size=ARGS.decode_batch_size,
        prefill_batch_size=ARGS.prefill_batch_size,
        decode_steps=ARGS.decode_steps,
        max_decode_length=ARGS.max_decode_length,
        use_prefill_attn_kernel=not ARGS.no_prefill_kernel,
    )

    shutdown_signal = threading.Event()
    serve_loop.serve_forever(shutdown_signal)

    serving.run_http_server(
        serve_loop,
        partial(tokenizer_encode, tokenizer),
        partial(tokenizer_decode, tokenizer),
        ARGS.server,
        shutdown_signal=shutdown_signal,
    )


if __name__ == "__main__":
    main()

########################################################################################################################
