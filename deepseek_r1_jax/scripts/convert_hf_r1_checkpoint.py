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
import json
import gzip
from pathlib import Path

import jax
from jax.sharding import PartitionSpec as P

from deepseek_r1_jax.model import ShardingRules, Config
from deepseek_r1_jax import chkpt_utils as utils

def main():
    root_path = Path("/mnt/storage/DeepSeek-R1")
    dest_path = Path("/mnt/storage/deepseek-r1-jax-chkpt")

    cfg = Config()
    cfg.quantize_mlp = False
    cfg.quantize_attn = True
    cfg.quantize_moe = True

    rules = ShardingRules(*(None for _ in dataclasses.fields(ShardingRules)))  # fully replicated
    cfg = dataclasses.replace(cfg, mesh=jax.make_mesh((1,), P("x")), rules=rules)
    params_map = json.loads(gzip.decompress((Path(__file__).parent.absolute()
                                             / "r1_hf_ckpt_params_map.json.gz").read_bytes()))
    utils.convert_hf_checkpoint(params_map, root_path, dest_path, cfg)

if __name__ == "__main__":
    main()
