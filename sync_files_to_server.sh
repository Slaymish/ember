#!/bin/bash

# Configuration
LOCAL_DIR="data/"                           # Local directory to sync
SERVER_USER="burkehami"                # Username on the GPU server
GPU_SERVER="cuda15.ecs.vuw.ac.nz"                    # IP/Hostname of the GPU server
JUMP_HOST_USER="burkehami"      # Username for the jump host
JUMP_HOST="barretts.ecs.vuw.ac.nz"                      # Jump host IP/Hostname
SERVER_DIR="/local/scratch/burkehami/data/"            # Directory on GPU server to sync to

# Rsync command with ProxyJump
echo "Starting rsync sync to GPU server via jump host..."

rsync -avz --progress -e "ssh -J ${JUMP_HOST_USER}@${JUMP_HOST}" "${LOCAL_DIR}/" "${SERVER_USER}@${GPU_SERVER}:${SERVER_DIR}"

if [ $? -eq 0 ]; then
    echo "Sync completed successfully!"
else
    echo "Error occurred during sync."
    exit 1
fi
