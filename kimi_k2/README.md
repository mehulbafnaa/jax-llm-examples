# Minimal Kimi K2 inference

**tl;dr: open-source Kimi K2 inference using JAX, minimal yet performant**

This model is a work in progress.

<br/>

This is a pure JAX implementation of Kimi K2 for inference, including a
checkpoint converter for the K2 Instruct weights. on TPU.
It should work on GPU.

The entire model is defined in [model.py](kimi_k2_jax/model.py) and invoked
via [main.py](main.py). Among other things, the model code demonstrates:
* an MLA attention implementation;
* expert and tensor-parallelism via JAX's
  [`shard_map`](https://docs.jax.dev/en/latest/sharded-computation.html#manual-parallelism-with-shard-map)
  for easy multi-device/multi-host computation; and
* simple int8 quantization.

## Quickstart

Due to the large model size (1T parameters), a multi-host platform is required to run
the full model.

Run on all hosts in the TPU cluster:
```
$ python3 main.py
```
e.g. for Cloud TPU:
```
$ gcloud compute tpus tpu-vm ssh {TPU_NAME} --worker=all \
    --command="cd ~/jax-llm-examples/kimi_k2 && python3 main.py"
```
