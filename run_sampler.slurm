#!/bin/bash
#SBATCH --job-name=process_fields    # Job name
#SBATCH --output=sampler_%j.out      # Output file name (%j = job ID)
#SBATCH --error=sampler_%j.err       # Error file name
#SBATCH --partition=volta            # Specify the partition
#SBATCH --time=01:00:00             # Maximum runtime in HH:MM:SS
#SBATCH --mem=32G                    # Memory per node

# Create a temporary directory for individual results
TEMP_DIR="temp_results_${SLURM_JOB_ID}"
mkdir -p $TEMP_DIR

# Initialize the combined results file
touch combined_results.txt

# Process each output file
for file in output/output_*.csv; do
    # Extract frequency from filename (e.g., 2.07e+14 from output_2.07e+14_3.00e+01nm_seed4.csv)
    if [[ $file =~ output_([0-9]+\.[0-9]+e\+[0-9]+)_ ]]; then
        freq="${BASH_REMATCH[1]}"
        # Remove decimal point and e+ to match the format sample_fields expects
        freq=$(echo $freq | tr -d '.' | sed 's/e+//')
        
        # Run sample_fields and append output directly to combined results
        ./sampler/sample_fields "$file" "$freq" >> combined_results.txt
    else
        echo "Warning: Couldn't extract frequency from filename: $file" >&2
    fi
done

# Clean up temporary directory
rm -rf $TEMP_DIR