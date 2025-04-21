import os
import re
import subprocess
import sys

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

for fname in sorted(os.listdir(output_dir)):
    if not fname.endswith(".csv"):
        continue
    
    params = extract_params_from_filename(fname)
    if params is None:
        print(f"Skipping unrecognized filename: {fname}")
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
        # print(f"{{\"frequency\": {params['frequency']}, \"disorder\": {params['disorder']}, \"seed\": {params['seed']}, \"flux\": {flux}}}")
        print(f"({params['frequency']}, {params['disorder']}, {params['seed']}, {flux}),")