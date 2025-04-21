#!/bin/bash
#SBATCH --job-name=dda_multi_gpu    # Job name
#SBATCH --output=dda_%A_%a.out      # Output file name (%A = job ID, %a = array index)
#SBATCH --error=dda_%A_%a.err       # Error file name
#SBATCH --array=0-9                 # Array jobs for 10 GPUs (0-9)
#SBATCH --partition=volta           # Specify the partition
#SBATCH --time=24:00:00             # Maximum runtime in HH:MM:SS
#SBATCH --mem=32G                   # Memory per node

# Calculate the seed based on array task ID
# Using array task ID + 1 to avoid seed=0
SEED=$((SLURM_ARRAY_TASK_ID + 1))

# Set disorder value (in nanometers)
DISORDER=30

# Run the program with the calculated seed
./solver $DISORDER $SEED
