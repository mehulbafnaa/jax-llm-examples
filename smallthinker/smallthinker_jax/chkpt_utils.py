import jax.numpy as jnp
import torch
from safetensors import safe_open
from tqdm import tqdm

from . import model as smallthinker_jax


def load_and_format_torch_weights(files, cfg):
    model = {}
    for file in tqdm(files):
        with safe_open(file, framework="torch") as f:
            for key in tqdm(f.keys(), leave=False):
                model[key] = f.get_tensor(key)
    return model


def convert_model_or_layer(weights, torch_weights, cfg, sequential=False):
    if sequential:
        raise NotImplementedError("Sequential conversion not yet supported.")

    def unpermute(w, num_heads, axis=0):
        return w.reshape(w.shape[0], num_heads, cfg.head_dim).swapaxes(axis, axis + 1).reshape(w.shape)

    for i in range(cfg.num_layers):
        # Attention weights
        weights.layers[i].q.quant = jnp.asarray(unpermute(torch_weights[f"model.layers.{i}.self_attn.q_proj.weight"].T, cfg.q_heads).to(torch.float32).numpy())
        weights.layers[i].k.quant = jnp.asarray(unpermute(torch_weights[f"model.layers.{i}.self_attn.k_proj.weight"].T, cfg.kv_heads).to(torch.float32).numpy())
        weights.layers[i].v.quant = jnp.asarray(unpermute(torch_weights[f"model.layers.{i}.self_attn.v_proj.weight"].T, cfg.kv_heads).to(torch.float32).numpy())
        weights.layers[i].o.quant = jnp.asarray(torch_weights[f"model.layers.{i}.self_attn.o_proj.weight"].T.to(torch.float32).numpy())

        # MoE weights
        weights.layers[i].moe_router.quant = jnp.asarray(torch_weights[f"model.layers.{i}.block_sparse_moe.primary_router.weight"].T.to(torch.float32).numpy())
        for j in range(cfg.moe_num_experts):
            weights.layers[i].moe_layers[j].gate.quant = jnp.asarray(torch_weights[f"model.layers.{i}.block_sparse_moe.experts.{j}.gate.weight"].T.to(torch.float32).numpy())
            weights.layers[i].moe_layers[j].up.quant = jnp.asarray(torch_weights[f"model.layers.{i}.block_sparse_moe.experts.{j}.up.weight"].T.to(torch.float32).numpy())
            weights.layers[i].moe_layers[j].down.quant = jnp.asarray(torch_weights[f"model.layers.{i}.block_sparse_moe.experts.{j}.down.weight"].T.to(torch.float32).numpy())

        # RMSNorm weights
        weights.layers[i].attn_pre_gamma = jnp.asarray(torch_weights[f"model.layers.{i}.input_layernorm.weight"].to(torch.float32).numpy())
        weights.layers[i].attn_post_gamma = jnp.asarray(torch_weights[f"model.layers.{i}.post_attention_layernorm.weight"].to(torch.float32).numpy())

    # Final layer weights
    weights.embedding = jnp.asarray(torch_weights["model.embed_tokens.weight"].to(torch.float32).numpy())
    weights.gamma_final = jnp.asarray(torch_weights["model.norm.weight"].to(torch.float32).numpy())
    weights.lm_head = jnp.asarray(torch_weights["lm_head.weight"].T.to(torch.float32).numpy())

    return weights
