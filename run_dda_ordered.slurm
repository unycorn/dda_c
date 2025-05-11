#!/bin/bash
#SBATCH --job-name=ordered_dda      # Job name
#SBATCH --output=dda_%j.out         # Output file name (%j expands to jobID)
#SBATCH --error=dda_%j.err          # Error file name
#SBATCH --partition=volta           # Specify the partition
#SBATCH --gpus=1                    # Request 1 GPU
#SBATCH --time=24:00:00             # Maximum runtime in HH:MM:SS
#SBATCH --mem=32G                   # Memory per node
#SBATCH --qos=normal                # Quality of Service

# Run the solver with no disorder
./solver 0 0 0 1