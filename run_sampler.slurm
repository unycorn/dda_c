#!/bin/bash
#SBATCH --job-name=process_fields    # Job name
#SBATCH --output=sampler_%A_%a.out   # Output file name (%A = job ID, %a = array index)
#SBATCH --error=sampler_%A_%a.err    # Error file name
#SBATCH --partition=volta            # Specify the partition
#SBATCH --array=0-7                  # Array jobs for 8 GPUs (volta has 8 nodes)
#SBATCH --time=01:00:00             # Maximum runtime in HH:MM:SS
#SBATCH --mem=32G                    # Memory per node

# Set which GPU this job will use (based on array task ID)
export CUDA_VISIBLE_DEVICES=$SLURM_ARRAY_TASK_ID

# Create a directory for partial results
mkdir -p partial_results

# Get list of files to process and calculate which ones this task should handle
files=(output/output_*.csv)
num_files=${#files[@]}
files_per_task=$(( (num_files + SLURM_ARRAY_TASK_COUNT - 1) / SLURM_ARRAY_TASK_COUNT ))
start_idx=$(( SLURM_ARRAY_TASK_ID * files_per_task ))
end_idx=$(( start_idx + files_per_task - 1 ))

# Process this task's subset of files
for (( i=start_idx; i<=end_idx && i<num_files; i++ )); do
    file="${files[$i]}"
    if [[ $file =~ output_([0-9]+\.[0-9]+e\+[0-9]+)_ ]]; then
        # Extract the frequency and pass it directly without modification
        freq="${BASH_REMATCH[1]}"
        
        # Run sample_fields and save to a partial results file
        ./sampler/sample_fields "$file" "$freq" > "partial_results/result_${SLURM_ARRAY_TASK_ID}_$(basename "$file")"
    else
        echo "Warning: Couldn't extract frequency from filename: $file" >&2
    fi
done

# If this is the last array task, combine all partial results
if [ $SLURM_ARRAY_TASK_ID -eq $((SLURM_ARRAY_TASK_COUNT-1)) ]; then
    # Wait for all other tasks to finish
    sleep 10
    
    # Combine all partial results into final output
    cat partial_results/result_* > combined_results.txt
    
    # Clean up partial results
    rm -rf partial_results
fi