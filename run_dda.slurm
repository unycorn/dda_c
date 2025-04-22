#!/bin/bash
#SBATCH --job-name=dda_multi_gpu    # Job name
#SBATCH --output=dda_%A_%a.out      # Output file name (%A = job ID, %a = array index)
#SBATCH --error=dda_%A_%a.err       # Error file name
#SBATCH --array=0-10                # Array jobs for 20 seed values
#SBATCH --partition=volta           # Specify the partition
#SBATCH --time=24:00:00             # Maximum runtime in HH:MM:SS
#SBATCH --mem=32G                   # Memory per node

# Calculate the seed based on array task ID
# Using array task ID + 1 to avoid seed=0
SEED=$((SLURM_ARRAY_TASK_ID + 1))

# Set disorder value (in nanometers)
# Run the program with the calculated seed

DISORDER=0
./solver $DISORDER $SEED

DISORDER=15
./solver $DISORDER $SEED

DISORDER=30
./solver $DISORDER $SEED

DISORDER=45
./solver $DISORDER $SEED

DISORDER=60
./solver $DISORDER $SEED

DISORDER=75
./solver $DISORDER $SEED

DISORDER=90
./solver $DISORDER $SEED

DISORDER=105
./solver $DISORDER $SEED

DISORDER=130
./solver $DISORDER $SEED

DISORDER=145
./solver $DISORDER $SEED

DISORDER=160
./solver $DISORDER $SEED