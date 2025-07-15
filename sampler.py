import os
import re
import subprocess
import argparse

# Parse command-line argument
parser = argparse.ArgumentParser()
parser.add_argument("parent_dir", help="Path to the directory containing parent CSVs and folders")
args = parser.parse_args()
parent_dir = args.parent_dir

# Ensure output logs folder exists
log_dir = os.path.join(parent_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

input_csvs = sorted([
    f for f in os.listdir(parent_dir)
    if f.startswith("cdm_input_") and f.endswith(".csv") and os.path.isfile(os.path.join(parent_dir, f))
])

for input_csv in input_csvs:
    print("input_csv", input_csv)
    input_path = os.path.join(parent_dir, input_csv)
    folder_name = f"{input_csv[:-4]}"
    folder_path = os.path.join(parent_dir, folder_name)

    if not os.path.isdir(folder_path):
        print(f"Skipping missing folder: {folder_path}")
        continue

    log_filename = f"freq_r_t.txt"
    log_path = os.path.join(folder_path, log_filename)

    while os.path.exists(log_path):
        print(log_path, "already generated")
        log_filename = f"{log_filename[:-4]}_augment.txt"
        log_path = os.path.join(folder_path, log_filename)

    with open(log_path, "w") as f:
        print(log_path, "log_path")
        for child_csv in os.listdir(folder_path):
            if not child_csv.endswith(".pols"):
                continue
            child_path = os.path.join(folder_path, child_csv)

            freq = float(re.search(r'output_freq_([0-9.]+e[+-]?[0-9]+)', child_path).group(1))
            #if freq < 350e12:
            #    continue

            cmd = ["/home/dharper/dda_c/sampler/sample_fields", input_path, child_path, child_path[-12:-4]]
            print(f"Running: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout, end='')
            f.write(result.stdout)