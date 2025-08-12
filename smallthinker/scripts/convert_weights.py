#!/usr/bin/env python3

import sys
from pathlib import Path
from argparse import ArgumentParser
import dataclasses
import shutil

def main(model_path: str | Path, ckpt_path: str | Path):
    try:
        from smallthinker_jax import model as smallthinker_jax
        from smallthinker_jax import chkpt_utils as utils
    except ImportError:
        sys.path.append(str(Path(__file__).parents[1].absolute()))

        from smallthinker_jax import model as smallthinker_jax
        from smallthinker_jax import chkpt_utils as utils

    from transformers import AutoConfig
    from safetensors import safe_open
    from tqdm import tqdm

    model_path, ckpt_path = Path(model_path).expanduser(), Path(ckpt_path).expanduser()
    files = list(model_path.glob("**/*safetensors"))
    assert len(files) > 1
    config_files = list(model_path.glob("**/config.json"))
    assert len(config_files) == 1, "Must have only one `config.json` file in the model path"
    config = AutoConfig.from_pretrained(config_files[0].parent, trust_remote_code=True)
    cfg = smallthinker_jax.smallthinker_to_jax_config(config)

    weights = smallthinker_jax.Weights.abstract(dataclasses.replace(cfg, quant_layer=False))

    if ckpt_path.exists():
        shutil.rmtree(ckpt_path)
        
    torch_weights = utils.load_and_format_torch_weights(files, cfg)
    converted_weights = utils.convert_model_or_layer(weights, torch_weights, cfg, sequential=False)
    smallthinker_jax.save_pytree(converted_weights, ckpt_path)

    additional_files = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    for additional_file in additional_files:
        full_paths = list(model_path.glob(f"**/{additional_file}"))
        if len(full_paths) != 1:
            print(f"Found more than 1 file for {additional_file}")
        if len(full_paths) == 0:
            continue
        full_path = full_paths[0]
        shutil.copyfile(full_path, ckpt_path / full_path.name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--source-path", default="../SmallThinker-4BA0.6B-Instruct", required=True, help="HF model directory path"
    )
    parser.add_argument(
        "--dest-path",
        default="./converted_checkpoint",
        required=True,
        help="JAX model model directory (to be created).",
    )
    args = parser.parse_args()
    main(args.source_path, args.dest_path)
