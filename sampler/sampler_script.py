import os
import re
import subprocess
import sys

if len(sys.argv) != 2:
    print("Usage: python sampler_script.py SEED")
    sys.exit(1)

command_line_arg_seed = int(sys.argv[1])

def extract_params_from_filename(filename):
    # Match formats like: output_(2.07e+14)_(3.00e+01nm)_(0.00e+00Hz)_(0.00e+00rad)_seed4.csv
    match = re.search(r"output_\(([0-9.]+e[+-]?[0-9]+)\)_\(([0-9.]+e[+-]?[0-9]+)nm\)_\(([0-9.]+e[+-]?[0-9]+)Hz\)_\(([0-9.]+e[+-]?[0-9]+)rad\)_seed([0-9]+)\.csv", filename)
    if match:
        return {
            'frequency': float(match.group(1)),
            'disorder': float(match.group(2)),
            'f0_disorder': float(match.group(3)),
            'angle_disorder': float(match.group(4)),
            'seed': int(match.group(5))
        }
    return None

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "output")

# Create output file for this seed
output_filename = f"sampler_results_seed{command_line_arg_seed}.txt"
with open(output_filename, 'w') as outfile:
    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith(".csv"):
            continue
        
        params = extract_params_from_filename(fname)
        if params is None:
            print(f"Skipping unrecognized filename: {fname}", file=sys.stderr)
            continue
        if params['seed'] != command_line_arg_seed:
            continue

        fullpath = os.path.join(output_dir, fname)
        # Run sample_fields and capture its output
        sample_fields_path = os.path.join(script_dir, "sample_fields")
        proc = subprocess.run([sample_fields_path, fullpath, str(params['frequency'])], 
                            capture_output=True, text=True)
        
        # Get both reflection and transmission coefficients from the output
        # Format is now (frequency,reflection,transmission)
        match = re.search(r"\((.*?),(.*?),(.*?)\)", proc.stdout)
        if match:
            reflection = float(match.group(2))
            transmission = float(match.group(3))
            outfile.write(f"({params['frequency']}, {params['disorder']}, {params['f0_disorder']}, "
                         f"{params['angle_disorder']}, {params['seed']}, {reflection}, {transmission}),\n")