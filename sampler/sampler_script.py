import os
import re
import subprocess
import sys

if len(sys.argv) != 2:
    print("Usage: python sampler_script.py SEED")
    sys.exit(1)

command_line_arg_seed = int(sys.argv[1])

def extract_params_from_filename(filename):
    # Match formats like: output_2.07e+14_3.00e+01nm_seed4.csv
    match = re.search(r"output_([0-9.]+e[+-]?[0-9]+)_([0-9.]+e[+-]?[0-9]+)nm_seed([0-9]+)\.csv", filename)
    if match:
        return {
            'frequency': float(match.group(1)),
            'disorder': float(match.group(2)),
            'seed': int(match.group(3))
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
        
        # Get the flux value from the output (which is in format "(frequency,flux),")
        match = re.search(r"\((.*?),(.*?)\)", proc.stdout)
        if match:
            flux = float(match.group(2))
            outfile.write(f"({params['frequency']}, {params['disorder']}, {params['seed']}, {flux}),\n")