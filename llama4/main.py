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
from etils import epath
import json
from pprint import pformat

import jax
from jax import numpy as jnp
from jax import random
from jax.sharding import set_mesh, AxisType, PartitionSpec as P
try:
    from jax.sharding import use_mesh as set_mesh
except ImportError:
    pass
import numpy as np

from llama4_jax import model as l4jax


def encode_input(tokenizer, texts, pad_id: int = l4jax.PAD_ID):
    assert isinstance(texts, list)
    inputs = [
        tokenizer.apply_chat_template([{"role": "user", "content": text}])
        + tokenizer.encode("<|header_start|>assistant<|header_end|>")
        for text in texts
    ]
    max_len = max([len(x) for x in inputs])
    inputs = [(max_len - len(x)) * [pad_id] + x for x in inputs]
    return np.array(inputs)


if __name__ == "__main__":
    jax.distributed.initialize()
    quant = True

    ckpt_path = epath.Path("~/bucket/Llama-4-Scout-Instruct").expanduser()
    if quant:
        ckpt_path = ckpt_path.parent / f"{ckpt_path.name}-quant"
    tokenizer = l4jax.load_tokenizer(ckpt_path / "tokenizer.json", ckpt_path / "tokenizer_config.json")

    mesh = jax.make_mesh(
        (1, 8, jax.device_count() // 8), ("x", "y", "z"), devices=jax.devices(), axis_types=(AxisType.Explicit,) * 3
    )
    cfg = l4jax.hf_to_jax_config(json.loads((ckpt_path / "config.json").read_text())["text_config"])
    cfg = dataclasses.replace(cfg, mesh=mesh, quant_attn=quant, quant_moe=quant, quant_mlp=quant)
    weights = l4jax.load_pytree(ckpt_path, l4jax.Weights.shardings(cfg))

    input = encode_input(
        tokenizer,
        [
            "Tell me your name",
            "What is the weather like expressed in long prose in Old English",
            "Do you like ice cream, be extremely precise",
        ],
    )

    with set_mesh(cfg.mesh):
        zero_cache = l4jax.KVCache.init(random.key(1), cfg, input.shape[0], cfg.max_seq_len)
        next_tokens, logits, cache = l4jax.prefill(input, weights, zero_cache, cfg)
        curr_tokens = next_tokens.at[:, cache.length - 1 : cache.length].get(out_sharding=P(None, None))
        tokens_list = []
        for _ in range(32):
            tokens_list.append(curr_tokens)
            curr_tokens, cache = l4jax.decode_step(curr_tokens, weights, cache, cfg)
        tokens = np.array(jnp.concatenate(tokens_list, axis=-1))
    responses = [tokenizer.decode(row) for row in tokens]
    print("Responses:")
    print(responses)
