#!/bin/bash
#SBATCH --job-name=morph_dda        # Job name
#SBATCH --output=dda_%A_%a.out      # Output file name (%A = job ID, %a = array index)
#SBATCH --error=dda_%A_%a.err       # Error file name
#SBATCH --array=0-10                # Array jobs for 11 seed values
#SBATCH --partition=volta           # Specify the partition
#SBATCH --gres=gpu:1                # Request 1 GPU using GRES syntax
#SBATCH --time=24:00:00             # Maximum runtime in HH:MM:SS
#SBATCH --mem=32G                   # Memory per node
#SBATCH --qos=normal                # Quality of Service

# Calculate the seed based on array task ID
# Using array task ID + 1 to avoid seed=0
SEED=$((SLURM_ARRAY_TASK_ID + 1))

# Define array of disorder values
DISORDERS=(5e12 15e12 25e12 35e12 45e12)

# Loop through disorder values
for DISORDER in "${DISORDERS[@]}"; do
    ./solver 0 $DISORDER 0 $SEED
done