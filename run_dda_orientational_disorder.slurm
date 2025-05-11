#!/bin/bash
#SBATCH --job-name=orient_dda      # Job name
#SBATCH --output=dda_%A_%a.out     # Output file name (%A = job ID, %a = array index)
#SBATCH --error=dda_%A_%a.err      # Error file name
#SBATCH --array=0-10               # Array jobs for 11 seed values
#SBATCH --partition=volta          # Specify the partition
#SBATCH --time=24:00:00           # Maximum runtime in HH:MM:SS
#SBATCH --mem=32G                 # Memory per node
#SBATCH --qos=normal             # Quality of Service

# Calculate the seed based on array task ID
SEED=$((SLURM_ARRAY_TASK_ID + 1))

# Define array of disorder values (in radians)
DISORDERS=(0.1 0.2 0.3 0.4 0.5)

# Loop through disorder values
for DISORDER in "${DISORDERS[@]}"; do
    ./solver 0 0 $DISORDER $SEED
done