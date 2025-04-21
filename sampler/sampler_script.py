import os
import re
import subprocess
import sys

def extract_frequency_from_filename(filename):
    # Match formats like: output_1.00e+14.csv or output_1.55e14.csv
    match = re.search(r"output_([0-9.]+e[+-]?[0-9]+)\\.csv", filename)
    if not match:
        # Also try without the escape on the dot (some systems are picky)
        match = re.search(r"output_([0-9.]+e[+-]?[0-9]+)\.csv", filename)
    return float(match.group(1)) if match else None

folder_path = "/home/dharper/dda_c/output/"
for fname in sorted(os.listdir(folder_path)):
    if not fname.endswith(".csv"):
        continue
    freq = extract_frequency_from_filename(fname)
    if freq is None:
        print(f"Skipping unrecognized filename: {fname}")
        continue

    fullpath = os.path.join(folder_path, fname)
    # print(f"Running ./sample_fields on {fname} with freq={freq:.2e} Hz")
    out = subprocess.run(["/home/dharper/dda_c/output/sampler/sample_fields", fullpath, str(freq)])
    # print(out)