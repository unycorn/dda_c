#!/bin/bash
#SBATCH --job-name=folders_dda      # Job name
#SBATCH --output=dda_%A_%a.out      # Output file name (%A = job ID, %a = array index)
#SBATCH --error=dda_%A_%a.err       # Error file name
#SBATCH --array=0-6                 # Array jobs for 7 GPU nodes
#SBATCH --partition=volta           # Specify the partition
#SBATCH --time=24:00:00            # Maximum runtime in HH:MM:SS
#SBATCH --mem=32G                   # Memory per node
#SBATCH --qos=normal               # Quality of Service

# Instead of manually setting CUDA_VISIBLE_DEVICES, let SLURM handle GPU assignment
# SLURM will set CUDA_VISIBLE_DEVICES automatically based on the allocated GPU

# Get the list of folders in csv_inputs directory
cd csv_inputs || exit 1
FOLDERS=(*)  # This will get all folders in csv_inputs
NUM_FOLDERS=${#FOLDERS[@]}

if [ $NUM_FOLDERS -eq 0 ]; then
    echo "No folders found in csv_inputs directory"
    exit 1
fi

# Calculate which folders this GPU should process
FOLDERS_PER_GPU=$(( (NUM_FOLDERS + 6) / 7 ))  # Ceiling division
START_IDX=$(( SLURM_ARRAY_TASK_ID * FOLDERS_PER_GPU ))
END_IDX=$(( START_IDX + FOLDERS_PER_GPU - 1 ))

# Print debug information
echo "Running on node: $(hostname)"
echo "GPU assignment: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi  # Show GPU status

# Process each folder assigned to this GPU
for (( i=START_IDX; i<=END_IDX && i<NUM_FOLDERS; i++ )); do
    FOLDER="${FOLDERS[i]}"
    echo "Processing folder: $FOLDER on GPU $SLURM_ARRAY_TASK_ID"
    cd "$FOLDER" || continue
    ../../solver . 351e12 400e12 12  # Adjust the command as needed
    cd ..
done