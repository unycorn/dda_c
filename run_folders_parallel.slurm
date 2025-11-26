#!/bin/bash
#SBATCH --job-name=freq_dda         # Job name
#SBATCH --output=dda_%A_%a.out      # Output file name (%A = job ID, %a = array index)
#SBATCH --error=dda_%A_%a.err       # Error file name
#SBATCH --array=0-7                 # Array jobs for 5 frequency chunks (adjust as needed)
#SBATCH --partition=ampere          # Specify the partition
#SBATCH --time=24:00:00             # Maximum runtime in HH:MM:SS
#SBATCH --mem=32G                   # Memory per node
#SBATCH --qos=normal                # Quality of Service

# Usage: sbatch run_folders_parallel.slurm <folder_name>
# Example: sbatch run_folders_parallel.slurm l0_p0_o0_m0

# Check if folder argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No folder specified"
    echo "Usage: sbatch $0 <folder_name>"
    exit 1
fi

FOLDER=$1

# Check if folder exists
if [ ! -d "csv_inputs/$FOLDER" ]; then
    echo "Error: Folder csv_inputs/$FOLDER does not exist"
    exit 1
fi

# Define frequency range parameters
FREQ_MIN=150e12  # 150 THz
FREQ_MAX=400e12  # 400 THz
TOTAL_FREQS=250
NUM_CHUNKS=7     # Number of frequency chunks (should match array size)

# Calculate frequency range for this job
FREQS_PER_CHUNK=$(( (TOTAL_FREQS + NUM_CHUNKS - 1) / NUM_CHUNKS ))  # Ceiling division
START_FREQ_IDX=$(( SLURM_ARRAY_TASK_ID * FREQS_PER_CHUNK ))
END_FREQ_IDX=$(( START_FREQ_IDX + FREQS_PER_CHUNK ))

# Ensure we don't exceed total frequencies
if [ $END_FREQ_IDX -gt $TOTAL_FREQS ]; then
    END_FREQ_IDX=$TOTAL_FREQS
fi

# Calculate actual frequency values
FREQ_RANGE=$(echo "$FREQ_MAX - $FREQ_MIN" | bc -l)
FREQ_STEP=$(echo "$FREQ_RANGE / ($TOTAL_FREQS - 1)" | bc -l)
START_FREQ=$(echo "$FREQ_MIN + $START_FREQ_IDX * $FREQ_STEP" | bc -l)
END_FREQ=$(echo "$FREQ_MIN + ($END_FREQ_IDX - 1) * $FREQ_STEP" | bc -l)
NUM_FREQS=$(( END_FREQ_IDX - START_FREQ_IDX ))

# Print debug information
echo "Running on node: $(hostname)"
echo "GPU assignment: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Processing folder: $FOLDER"
echo "Frequency chunk: $SLURM_ARRAY_TASK_ID"
echo "Frequency indices: $START_FREQ_IDX to $(( END_FREQ_IDX - 1 ))"
printf "Frequency range: %.3e Hz to %.3e Hz\n" $START_FREQ $END_FREQ
echo "Number of frequencies: $NUM_FREQS"
nvidia-smi  # Show GPU status

# Navigate to the folder and run solver
cd csv_inputs/$FOLDER || exit 1
../../solver . $START_FREQ $END_FREQ $NUM_FREQS