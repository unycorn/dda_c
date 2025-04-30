#!/bin/bash
#SBATCH --job-name=dda_multi_gpu    # Job name
#SBATCH --output=dda_%A_%a.out      # Output file name (%A = job ID, %a = array index)
#SBATCH --error=dda_%A_%a.err       # Error file name
#SBATCH --array=0-10                # Array jobs for 11 seed values
#SBATCH --partition=volta           # Specify the partition
#SBATCH --time=24:00:00             # Maximum runtime in HH:MM:SS
#SBATCH --mem=32G                   # Memory per node

# Calculate the seed based on array task ID
# Using array task ID + 1 to avoid seed=0
SEED=$((SLURM_ARRAY_TASK_ID + 1))

# Set disorder value (in nanometers)
# Run the program with the calculated seed

# solver usage: ./solver <disorder_nm> <f0_disorder_Hz> <angle_disorder_rad> <seed>
# disorder types:        positional    morphological    orientational

# Define array of disorder values
DISORDERS=(5e12 15e12 25e12 35e12 45e12)

# Loop through disorder values
for DISORDER in "${DISORDERS[@]}"; do
    ./solver 0 $DISORDER 0 $SEED
done