#!/bin/bash
#SBATCH --job-name=sampler_seeds    # Job name
#SBATCH --output=sampler_%A_%a.out  # Output file name (%A = job ID, %a = array index)
#SBATCH --error=sampler_%A_%a.err   # Error file name
#SBATCH --array=1-11                # Array jobs for seeds 1-11
#SBATCH --partition=volta           # Specify the partition
#SBATCH --time=01:00:00            # Maximum runtime in HH:MM:SS
#SBATCH --mem=32G                  # Memory per node

# Get the seed from the array task ID (no need to add 1 since we start array at 1)
SEED=$SLURM_ARRAY_TASK_ID

# Set which GPU this job will use (mod 8 to wrap around since we only have 8 GPUs)
# export CUDA_VISIBLE_DEVICES=$((($SLURM_ARRAY_TASK_ID-1) % 8))

# Run the sampler script with this seed
/shared/Anaconda3-2020.02/bin/python3 sampler/sampler_script.py $SEED