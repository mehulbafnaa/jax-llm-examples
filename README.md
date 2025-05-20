# JAX LLM examples

A collection (in progress) of example high-performance large language model
implementations, written with JAX.

Current contents include:

* [DeepSeek R1](deepseek_r1_jax/)
* [Llama 4](llama4/)
* [Llama 3](llama3/)
* [Qwen 3](qwen3/)

## Working with Multi-Host Clusters

When working with many accelerators, JAX offers
[Distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
with a global view of the computation, but the program needs to be run on many hosts
each of which controls a subset of the actual accelerators.

The simplest way to run a JAX program on multiple hosts is to [run the same
Python file from all the hosts at the same
time](https://docs.jax.dev/en/latest/multi_process.html) - for example by
launching an ssh command on all hosts in the cluster.

However, for development it's often easier to (1.) efficiently share code
changes to all hosts, (2.) have a way of easily launching computation on all
hosts and (3.) have the ability to debug interactively.

This section shows how you can do that:
1. Shared disk setup - [NFS](https://ubuntu.com/server/docs/network-file-system-nfs) & [gcsfuse](https://github.com/GoogleCloudPlatform/gcsfuse)
2. [Batch SSH commands](https://cloud.google.com/sdk/gcloud/reference/alpha/compute/tpus/tpu-vm/ssh)
3. Interactive cluster setup with [ipyparallel](https://ipyparallel.readthedocs.io/en/latest/)

This guide has specific instructions for setting up a TPU Pod with GCS, but a
similar setup can be applied to any Linux multi-host platform, including GPU.

### Creating a Multi-Host TPU VM

```bash
TPU_ZONE="zone, e.g. us-central1-a"
PROJECT="your-project"
IMAGE="v2-alpha-tpuv5-lite"
ACCELERATOR="v5litepod-64"
TPU_NAME="my_tpu"
TPU_NAME="$NAME_PREFIX"-"$ACCELERATOR"

gcloud alpha compute tpus tpu-vm create "$TPU_NAME" --zone="$TPU_ZONE" \
  --project="$PROJECT" --accelerator-type="$ACCELERATOR" --version="$IMAGE"
```

### Utilities and Package Installation

```bash
# Util function to start remote processes on TPU machines.
TPU_NAME="..."
TPU_ZONE="..."
TPU_PROJECT="..."

tpu_exec() {
    local workers=$(seq $1 $2 | tr '\n' ',')
    gcloud alpha compute tpus tpu-vm ssh --zone="$TPU_ZONE" --project="$TPU_PROJECT" \
      "$TPU_NAME" --worker="$workers" --command="$3"
}
```

```bash
# Install required packages and virtualenv.
INSTALL_COMMAND=$(cat << EOM
  sudo apt update
  sudo apt install -y nfs-common nfs-kernel-server nfs-server net-tools tmux python3-ipyparallel
  curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
  uv python install 3.10 && uv venv --python 3.10
  cd ~/
  uv pip install -U "jax[tpu]" ipyparallel 
  if [ ! -d jax-llm-examples ]; then
    git clone https://github.com/jax-ml/jax-llm-examples.git
  fi
  cd jax-llm-examples
  uv pip install -e .
EOM
)

tpu_exec 0 15 "$INSTALL_COMMAND"
```

### Setting up Code & Data on the TPU-VM

#### 1. [gcsfuse](https://cloud.google.com/storage/docs/cloud-storage-fuse/install#install-source-code)

For datasets and checkpoints.

```bash
mkdir {local_folder}
gcsfuse --implicit-dirs {bucket_name_no_gs://} {local_folder}
```

#### 2. NFS

For code consistency between hosts in the TPU Pod / Cluster.

```bash
# on worker 0
WORKER0_IP="..." # Internal IP address
mkdir -p ~/nfs; sudo umount ~/nfs
echo "$HOME/nfs $WORKER0_IP/24(rw,sync,no_subtree_check)" | sudo tee /etc/exports
sudo exportfs -a
sudo systemctl enable nfs-server; sudo systemctl restart nfs-server
sudo chown $USER:$USER -R ~/nfs
```

```bash
MOUNT_COMMAND=$(cat << EOM
  mkdir -p ~/nfs
  sudo umount ~/nfs
  # VM_USER should be username in your TPU VM and should be the same across all VM workers.
  sudo mount -t nfs WORKER0_IP:/home/VM_USER/nfs ~/nfs
EOM
)
tpu_exec 1 15 "$MOUNT_COMMAND"
```

#### (Optionally) 3. [sshfs](https://github.com/libfuse/sshfs)

For a quick preview from a local machine.

```bash
sshfs ~/local_folder TPU_WORKER_0_IP:~/remote_folder
```

### Starting the `ipyparallel` Cluster

Start $N - 1$ workers (ipyparallel calls them `engines`) because we want worker 0 to execute interactively.

```bash
SERVER_IP="..."
CONTROLLER_SETUP=$(cat << EOM
tmux kill-session -t controller; pkill -9 python
tmux new -d -s controller '\
  . ~/.venv/bin/activate && ipcontroller --profile-dir=~/nfs --ip=$SERVER_IP'
EOM
)

ENGINE_SETUP=$(cat << EOM
tmux kill-session -t engine; pkill -9 ipengine
tmux new -d -s engine '. ~/.venv/bin/activate && ipengine --profile-dir=~/nfs'
EOM
)

tpu_exec 0 0  "$CONTROLLER_SETUP"  # only worker 0
tpu_exec 1 15 "$ENGINE_SETUP" # all workers except worker 0
```

#### Confirm ipyparallel setup works by ssh'ing into worker 0.

> Cell 0:
```python
import ipyparallel as ip
from pathlib import Path
connection_file = Path("~/nfs/security/ipcontroller-client.json").expanduser()
client = ip.Client(connection_info=connection_file)
print(sorted(list(client._engines.keys())))  # one less than worker num
# this process is the final worker
```

> Cell 1:
```python
%%px --local
import socket
import jax
jax.distributed.initialize()  # no arguments, TPUs automatically detect peers
print(f"Hello from {socket.gethostname()}")
```

> Note: "--local" argument means "also run on this process", it's necessary to
> get easy access to the output of computations on worker 0

### Troubleshooting

- Running into import errors/package incompatibility errors when running on the engines.\
    Solution: check if you have activated your virtualenv before running jupyter and on all of the engines.\
    E.g.\
    ```python
    %%px --local
    print(sys.executable)
    ```
- Your notebook is hanging and eventually times out when running jax.distributed.initialize().\
    Solution: you are likely running a pre-existing jupyter session and have not cleared the Engines.\
    Run `client[:].abort()` to clear.\
    Worst case you may need to restart the engines via:\
    `tpu_exec 0 0 "$CONTROLLER_CMD"`.
- You've encountered OOM on a run that shouldn't have run out of memory.\
    Solution: you have again likely not cleared pre-existing sessions and still have weights loaded in memory.\
    Please run the above suggestions and try again.
