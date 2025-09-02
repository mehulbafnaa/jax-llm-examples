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

import dataclasses
import threading
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import serving_jax as serving
from jax import random
from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P
from serving_jax import attention_cache_utils

from deepseek_r1_jax import chkpt_utils as dsjax_utils
from deepseek_r1_jax import model as dsjax


jax.config.update("jax_explain_cache_misses", True)
# jax.config.update("jax_compilation_cache_dir", str(Path("~/.cache/jax").expanduser()))
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)


def encode_input(tokenizer, texts, pad_id: int = 0):
    assert isinstance(texts, list)
    inputs = [
        tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True) for text in texts
    ]
    max_len = max([len(x) for x in inputs])
    return np.array([(max_len - len(x)) * [pad_id] + x for x in inputs])


tokenizer_encode = lambda tokenizer, text: encode_input(tokenizer, [text])[0].tolist()
tokenizer_decode = lambda tokenizer, tokens: tokenizer.decode(tokens)


def distributed_init():
    # for TPU
    jax.distributed.initialize()

    # for GPU/CPU
    # process_idx = int(socket.gethostname().split("-")[-1]) - 1  # a scheme where hosts are (host-1, host-2, ...)
    # jax.distributed.initialize(os.environ["COORDINATOR_ADDRESS"], 2, process_idx)
    # jax.distributed.initialize()


def load_model():
    parser = ArgumentParser()
    parser.add_argument("--server", action="store_true", help="Make this node the main server.", default=False)
    ARGS = parser.parse_args()

    distributed_init()
    devices = jax.devices()  # this helps catch distributed errors quickly

    ckpt_path = Path(f"~/bucket/deepseek-r1-jax-chkpt").expanduser()
    tokenizer = dsjax.load_tokenizer()
    assert ckpt_path.is_dir()
    print("---> Model config loaded")

    mesh = jax.make_mesh((1, 8, len(devices) // 8), P("x", "y", "z"), devices=devices, axis_types=(AxisType.Auto,) * 3)
    cfg = dataclasses.replace(dsjax.Config(), max_seq_len=1024, mesh=mesh)#, num_layers=4)
    weights = dsjax_utils.load_model(ckpt_path, cfg)
    decode_weights, prefill_weights = weights, weights

    print("---> Weights loaded")
    serve_cfg = serving.ServingConfig(
        decode_steps=32, max_decode_length=64, decode_batch_size=8, prefill_batch_size=1, prefix_chunk_size=64, max_ondevice_buffers=16
    )
    decode_cache = serving.AttentionWrapper(
        dsjax.KVCache.init(random.key(0), cfg, serve_cfg.decode_batch_size, cfg.max_seq_len),
        attention_cache_utils.kvcache_get_sequence,
        attention_cache_utils.kvcache_insert_sequences
    )
    prefill_cache = serving.AttentionWrapper(
        dsjax.KVCache.init(random.key(0), cfg, serve_cfg.prefill_batch_size, cfg.max_seq_len),
        attention_cache_utils.kvcache_get_sequence,
        attention_cache_utils.kvcache_insert_sequences
    )

    sampler = partial(jnp.argmax, axis=-1)

    @partial(jax.jit, donate_argnames=("cache",))
    def forward_fn(inputs, weights, cache, cfg):
        logits, cache = dsjax.forward(inputs, (inputs != dsjax.PAD_ID).astype(np.int32), weights, cfg, cache)
        return sampler(logits), cache

    serve_loop = serving.ServingLoop(
        serve_cfg, cfg, forward_fn, prefill_weights, prefill_cache, decode_weights, decode_cache, ARGS.server
    )
    print("---> Created the serving loop")

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
    load_model()

########################################################################################################################
