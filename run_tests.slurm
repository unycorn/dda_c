#!/bin/bash
#SBATCH --job-name=dda_tests    # Job name
#SBATCH --output=tests_%j.out   # Output file name (%j expands to jobID)
#SBATCH --error=tests_%j.err    # Error file name
#SBATCH --partition=volta       # Specify the partition
#SBATCH --gres=gpu:1           # Request 1 GPU using GRES syntax
#SBATCH --time=00:30:00        # Maximum runtime in HH:MM:SS
#SBATCH --mem=8G               # Memory per node
#SBATCH --qos=normal          # Quality of Service

# Load necessary modules (adjust these based on your cluster setup)
# module load cuda
# module load cmake

# Set CUDA paths if needed
export CUDA_PATH=/usr/local/cuda  # Adjust this path for your cluster

# Build and run tests
make clean
make test