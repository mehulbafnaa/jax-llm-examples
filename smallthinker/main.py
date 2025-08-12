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
from pprint import pprint

import jax
from jax import numpy as jnp
from jax import random
from jax.sharding import use_mesh, AxisType, PartitionSpec as P
import numpy as np

from smallthinker_jax import model as smallthinker_jax


def encode_input(tokenizer, texts: list[str], pad_id: int = 0):
    assert isinstance(texts, list)
    inputs = [
        tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
        for text in texts
    ]
    max_len = max([len(x) for x in inputs])
    inputs = [(max_len - len(x)) * [pad_id] + x for x in inputs]
    return np.array(inputs)


if __name__ == "__main__":
    print("Starting SmallThinker inference...")
    
    # Try to initialize distributed JAX for multi-host TPU
    try:
        jax.distributed.initialize()
        print("Multi-host JAX initialized")
    except ValueError as e:
        print(f"Single host mode (multi-host failed: {e})")
    
    quant = False  # Checkpoint was saved with quantized weights (.quant attributes)
    print(f"Quantization: {quant}")

    print("Loading tokenizer...")
    ckpt_path = epath.Path("/home/mehulbafna/nfs_share/axam/jax-llm-examples/smallthinker/converted_checkpoint")
    tokenizer = smallthinker_jax.load_tokenizer(ckpt_path / "tokenizer.json", ckpt_path / "tokenizer_config.json")
    print("Tokenizer loaded")

    devices = jax.devices()
    device_count = len(devices)
    print(f"Available devices: {device_count}, devices: {devices}")
    
    # TPU v4-8 has 8 devices total (4 per host × 2 hosts)
    # Original mesh: (1, 8, jax.device_count() // 8) = (1, 8, 1) for v4-8
    mesh = jax.make_mesh(
        (1, 8, device_count // 8), ("x", "y", "z"), devices=jax.devices(), 
        axis_types=(AxisType.Explicit,) * 3
    )
    print("Loading config...")
    cfg = smallthinker_jax.load_config(ckpt_path / "config.json")
    cfg = dataclasses.replace(cfg, mesh=mesh, quant_layer=quant, quant_cache=quant)
    print("Config loaded, loading weights...")
    weights = smallthinker_jax.load_pytree(ckpt_path, smallthinker_jax.Weights.shardings(cfg))
    print("Weights loaded!")
    
    # Debug: Check if weights loaded properly
    print(f"Weight layers count: {len(weights.layers) if hasattr(weights, 'layers') else 'No layers'}")
    if hasattr(weights, 'layers') and len(weights.layers) > 0:
        layer0 = weights.layers[0]
        print(f"Layer 0 q shape: {layer0.q.shape if layer0.q is not None else 'None'}")
        print(f"Layer 0 k shape: {layer0.k.shape if layer0.k is not None else 'None'}")
    else:
        print("Weights structure:", type(weights), dir(weights))

    input_texts = [
        "Tell me your name",
        "What is the weather like expressed in long prose in Old English",
        "Do you like ice cream, be extremely precise",
    ]
    input_tokens = encode_input(tokenizer, input_texts)

    with use_mesh(cfg.mesh):
        zero_cache = smallthinker_jax.KVCache.init(random.key(1), cfg, input_tokens.shape[0], cfg.max_seq_len)
        next_tokens, logits, cache = smallthinker_jax.prefill(input_tokens, weights, zero_cache, cfg)
        curr_tokens = next_tokens.at[:, cache.length - 1 : cache.length].get(out_sharding=P(None, None))
        tokens_list = []
        for _ in range(128): # Generate more tokens
            tokens_list.append(curr_tokens)
            curr_tokens, cache = smallthinker_jax.decode_step(curr_tokens, weights, cache, cfg)
        tokens = np.array(jnp.concatenate(tokens_list, axis=-1))
    
    responses = [tokenizer.decode(row) for row in tokens]
    print("Responses:")
    pprint(responses)
