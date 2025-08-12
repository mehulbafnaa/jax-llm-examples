#!/bin/bash

# Distributed SmallThinker inference script for TPU v4-8
# Run this script simultaneously on both hosts

# Set environment variables for JAX distributed initialization
export JAX_COORDINATOR_ADDRESS="t1v-n-0b35dafc-w-0:1234"
export JAX_COORDINATOR_PORT="1234"
export JAX_PROCESS_COUNT="2"

# Determine process ID based on hostname
if [[ $(hostname) == *"-w-0" ]]; then
    export JAX_PROCESS_ID="0"
    echo "Host 0 (coordinator): $(hostname)"
else
    export JAX_PROCESS_ID="1" 
    echo "Host 1 (worker): $(hostname)"
fi

echo "JAX_COORDINATOR_ADDRESS: $JAX_COORDINATOR_ADDRESS"
echo "JAX_PROCESS_COUNT: $JAX_PROCESS_COUNT"
echo "JAX_PROCESS_ID: $JAX_PROCESS_ID"

# Change to smallthinker directory and run inference
cd /home/mehulbafna/nfs_share/axam/jax-llm-examples/smallthinker
uv run main.py