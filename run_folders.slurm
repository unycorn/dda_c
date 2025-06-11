#!/bin/bash
#SBATCH --job-name=folders_dda      # Job name
#SBATCH --output=dda_%A_%a.out      # Output file name (%A = job ID, %a = array index)
#SBATCH --error=dda_%A_%a.err       # Error file name
#SBATCH --array=0-7                 # Array jobs for 8 GPU nodes
#SBATCH --partition=volta           # Specify the partition
#SBATCH --time=24:00:00            # Maximum runtime in HH:MM:SS
#SBATCH --mem=32G                   # Memory per node
#SBATCH --qos=normal               # Quality of Service

# Set which GPU this job will use
export CUDA_VISIBLE_DEVICES=$SLURM_ARRAY_TASK_ID

# Get the list of folders in csv_inputs directory
cd csv_inputs || exit 1
FOLDERS=(*)  # This will get all folders in csv_inputs
NUM_FOLDERS=${#FOLDERS[@]}

if [ $NUM_FOLDERS -eq 0 ]; then
    echo "No folders found in csv_inputs directory"
    exit 1
fi

# Calculate which folders this GPU should process
FOLDERS_PER_GPU=$(( (NUM_FOLDERS + 7) / 8 ))  # Ceiling division
START_IDX=$(( SLURM_ARRAY_TASK_ID * FOLDERS_PER_GPU ))
END_IDX=$(( START_IDX + FOLDERS_PER_GPU - 1 ))

# Process each folder assigned to this GPU
for (( i=START_IDX; i<=END_IDX && i<NUM_FOLDERS; i++ )); do
    FOLDER="${FOLDERS[i]}"
    echo "Processing folder: $FOLDER on GPU $SLURM_ARRAY_TASK_ID"
    cd "$FOLDER" || continue
    ../../solver .
    cd ..
done