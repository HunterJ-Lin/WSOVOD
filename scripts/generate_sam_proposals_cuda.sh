#!/usr/bin/env bash

# Assign the first argument to GPUS, the number of GPUs to use.
GPUS=$1

# Set the number of nodes, defaulting to 1 if not set.
NNODES=${NNODES:-1}

# Set the rank of the node, defaulting to 0 if not set.
NODE_RANK=${NODE_RANK:-0}

# Set the port, defaulting to a random value within a specific range if not set.
PORT=${PORT:-$((28500 + $RANDOM % 2000))}

# Set the master address, defaulting to localhost if not set.
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Check if torchrun is available.
if command -v torchrun &> /dev/null
then
  echo "Using torchrun mode."
  # Set environment variables for Python path and thread settings, then execute the Python script with torchrun.
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    torchrun --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${PORT} \
    --nproc_per_node=${GPUS} \
    tools/generate_sam_proposals_cuda.py "${@:2}"
else
  echo "Using launch mode."
  # Fallback to using python -m torch.distributed.launch if torchrun is not available.
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m torch.distributed.launch \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${PORT} \
    --nproc_per_node=${GPUS} \
    tools/generate_sam_proposals_cuda.py "${@:2}"
fi