#!/bin/bash
#SBATCH --job-name=dda_multi_gpu    # Job name
#SBATCH --output=dda_%A_%a.out      # Output file name (%A = job ID, %a = array index)
#SBATCH --error=dda_%A_%a.err       # Error file name
#SBATCH --array=0-7                 # Array jobs for 8 GPUs (volta partition)
#SBATCH --partition=volta           # Specify the partition
#SBATCH --time=24:00:00             # Maximum runtime in HH:MM:SS
#SBATCH --mem=32G                   # Memory per node

# Set which GPU this job will use (based on array task ID)
export CUDA_VISIBLE_DEVICES=$SLURM_ARRAY_TASK_ID

# Define disorder values
DISORDERS=(0 15 30 45 60 75 90 105 130 145 160)

# Calculate which disorder values this GPU should handle
# We'll use the array task ID to determine which disorder values to process
# This way each GPU processes a subset of the disorder values
NUM_DISORDERS=${#DISORDERS[@]}
CHUNK_SIZE=$(( (NUM_DISORDERS + SLURM_ARRAY_TASK_COUNT - 1) / SLURM_ARRAY_TASK_COUNT ))
START_IDX=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE - 1 ))

# Use array task ID + 1 as the seed
SEED=$((SLURM_ARRAY_TASK_ID + 1))

# Process this GPU's subset of disorder values
for (( i=START_IDX; i<=END_IDX && i<NUM_DISORDERS; i++ )); do
    DISORDER=${DISORDERS[$i]}
    echo "Processing disorder=$DISORDER with seed=$SEED on GPU $CUDA_VISIBLE_DEVICES"
    ./solver $DISORDER $SEED
done