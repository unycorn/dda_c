#!/usr/bin/env python3
"""
Extract existing absorption values from .pols files
and save them to a CSV file.
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd


def read_pols_file_with_absorption(filename):
    """
    Read a .pols file that already contains absorption.

    Returns:
        freq: Frequency in Hz
        absorption: Absorbed power in Watts
    """
    with open(filename, 'rb') as f:
        N = np.fromfile(f, dtype=np.int32, count=1)[0]
        freq = np.fromfile(f, dtype=np.float64, count=1)[0]

        # Skip polarization data
        f.seek(4 + 8 + 2 * N * 16)

        absorption = np.fromfile(f, dtype=np.float64, count=1)[0]

    return freq, absorption


def extract_absorption(folder_path, output_csv):
    pols_files = glob.glob(os.path.join(folder_path, "*.pols"))

    if not pols_files:
        print("No .pols files found.")
        return

    frequencies = []
    absorptions = []

    for pols_file in sorted(pols_files):
        try:
            freq, absorption = read_pols_file_with_absorption(pols_file)
            frequencies.append(freq)
            absorptions.append(absorption)
            print(f"{os.path.basename(pols_file)}  f={freq:.3e}  A={absorption:.6e}")
        except Exception as e:
            print(f"Skipping {pols_file}: {e}")

    df = pd.DataFrame({
        "Frequency_Hz": frequencies,
        "Absorption_W": absorptions
    }).sort_values("Frequency_Hz")

    df.to_csv(output_csv, index=False)
    print(f"\nSaved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract existing absorption values from .pols files"
    )
    parser.add_argument("pols_folder", help="Folder containing .pols files")
    parser.add_argument("output_csv", help="Output CSV file")

    args = parser.parse_args()

    if not os.path.isdir(args.pols_folder):
        print("Invalid folder.")
        sys.exit(1)

    extract_absorption(args.pols_folder, args.output_csv)


if __name__ == "__main__":
    main()
