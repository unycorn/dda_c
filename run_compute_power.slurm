#!/bin/bash
#SBATCH --job-name=compute_power_dda    # Job name
#SBATCH --output=power_%A_%a.out        # Output file name (%A = job ID, %a = array index)
#SBATCH --error=power_%A_%a.err         # Error file name
#SBATCH --array=0-30                    # Array jobs for processing folders
#SBATCH --partition=ampere              # Specify the partition
#SBATCH --time=24:00:00                # Maximum runtime in HH:MM:SS
#SBATCH --mem=16G                      # Memory per node (reduced since no GPU needed)
#SBATCH --cpus-per-task=4              # CPU cores for parallel processing

# Load required modules
module load python/3.9  # Adjust based on your system's module system
# module load anaconda3   # Alternative if using conda

# Get the list of folders in csv_inputs directory
cd csv_inputs || exit 1
FOLDERS=(*)  # This will get all folders in csv_inputs
NUM_FOLDERS=${#FOLDERS[@]}

if [ $NUM_FOLDERS -eq 0 ]; then
    echo "No folders found in csv_inputs directory"
    exit 1
fi

# Calculate which folders this task should process
FOLDERS_PER_TASK=$(( (NUM_FOLDERS + 30) / 31 ))  # Ceiling division for 31 array tasks (0-30)
START_IDX=$(( SLURM_ARRAY_TASK_ID * FOLDERS_PER_TASK ))
END_IDX=$(( START_IDX + FOLDERS_PER_TASK - 1 ))

# Print debug information
echo "Running on node: $(hostname)"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing folders $START_IDX to $END_IDX"
echo "Total folders: $NUM_FOLDERS"

# Process each folder assigned to this task
for (( i=START_IDX; i<=END_IDX && i<NUM_FOLDERS; i++ )); do
    FOLDER="${FOLDERS[i]}"
    echo "Processing folder: $FOLDER"
    
    # Check if folder exists and contains expected files
    if [ ! -d "$FOLDER" ]; then
        echo "Warning: Folder $FOLDER does not exist, skipping"
        continue
    fi
    
    cd "$FOLDER" || continue
    
    # Find the CSV file (should match pattern cdm_input_*.csv)
    CSV_FILE=$(ls cdm_input_*.csv 2>/dev/null | head -n 1)
    
    if [ -z "$CSV_FILE" ]; then
        echo "Warning: No cdm_input_*.csv file found in $FOLDER, skipping"
        cd ..
        continue
    fi
    
    # Extract the number from the CSV filename to find corresponding pols folder
    NUMBER=$(echo "$CSV_FILE" | sed 's/cdm_input_\(.*\)\.csv/\1/')
    POLS_FOLDER="cdm_input_$NUMBER"
    
    if [ ! -d "$POLS_FOLDER" ]; then
        echo "Warning: Polarization folder $POLS_FOLDER not found, skipping"
        cd ..
        continue
    fi
    
    echo "  CSV file: $CSV_FILE"
    echo "  Pols folder: $POLS_FOLDER"
    
    # Run the compute_power script
    # Adjust parameters as needed:
    # --incident_power: Set appropriate incident power value
    # --freq_target: Target frequency (default 220 THz)
    # --n_theta, --n_phi: Sampling resolution
    # --save_data: Save results to file
    
    OUTPUT_FILE="power_results_${NUMBER}.csv"
    
    python3 ../../analysis/DirectionalPowerMeasure/compute_power.py \
        "$CSV_FILE" \
        --incident_power 1.04238780e-13 \
        --n_theta 100 \
        --n_phi 200 \
        --save_data "$OUTPUT_FILE" \
        2>&1 | tee "compute_power_${NUMBER}.log"
    
    # Check if the script ran successfully
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "Successfully processed $FOLDER"
    else
        echo "Error processing $FOLDER - check compute_power_${NUMBER}.log"
    fi
    
    cd ..
done

echo "Task $SLURM_ARRAY_TASK_ID completed"