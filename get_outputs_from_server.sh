#!/bin/bash

# Configuration
SERVER_USER="burkehami"
GPU_SERVER="cuda15.ecs.vuw.ac.nz"
JUMP_HOST_USER="burkehami"
JUMP_HOST="barretts.ecs.vuw.ac.nz"
REMOTE_DIR="/local/scratch/burkehami/data/outputs"
LOCAL_DIR="models/"  # Local destination

# Rsync command to fetch models
echo "Fetching model files from GPU server..."
rsync -avz --progress -e "ssh -J ${JUMP_HOST_USER}@${JUMP_HOST}" \
    "${SERVER_USER}@${GPU_SERVER}:${REMOTE_DIR}" "${LOCAL_DIR}"

if [ $? -eq 0 ]; then
    echo "Files fetched successfully!"
else
    echo "Error occurred during fetch."
    exit 1
fi
