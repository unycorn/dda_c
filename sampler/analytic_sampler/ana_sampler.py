#!/usr/bin/env python3
import sys
import numpy as np
import os
from pathlib import Path
import csv

# Physical constants
EPSILON_0 = 8.8541878128e-12  # vacuum permittivity in F/m
SPEED_OF_LIGHT = 299792458.0  # speed of light in m/s

def read_polarizations_binary(filename):
    with open(filename, 'rb') as f:
        # Read N (4-byte integer)
        N = np.fromfile(f, dtype=np.int32, count=1)[0]
        
        # Read frequency (8-byte double)
        freq = np.fromfile(f, dtype=np.float64, count=1)[0]
        
        # Read complex doubles (2 components per point: Ex and Mz)
        data = np.fromfile(f, dtype=np.complex128, count=2*N)
        
        return N, freq, data.reshape(-1, 2)  # reshape to (N, 2) array

def calculate_r(freq, ex_sum, area):
    # Calculate vacuum wavenumber k = 2π/λ = 2πf/c
    k = 2 * np.pi * freq / SPEED_OF_LIGHT
    # Calculate r using the provided formula
    r = 1j * k / (2 * EPSILON_0 * area) * ex_sum
    return r

def calculate_R_T(r):
    # Calculate reflection coefficient R = |r|²
    R = np.abs(r) ** 2
    # Calculate transmission coefficient T = |1 + r|²
    T = np.abs(1 + r) ** 2
    return R, T

def process_pols_folder(folder_path, area):
    # Store results as (frequency, R, T) tuples
    results_x = []
    results_y = []

    csv_file_path = folder_path + '.csv'
    if os.path.exists(csv_file_path):
        thetas = []
        with open(csv_file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                thetas.append(float(row['theta']))
        thetas = np.array(thetas)
    else:
        print(f"CSV file {csv_file_path} not found. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    # Process all .pols files in the folder
    for file_path in sorted(Path(folder_path).glob('*.pols')):
        try:
            N, freq, data = read_polarizations_binary(str(file_path))

            e_localx_pol = data[:, 0] # electric polarization x-component in SRR's local frame
            e_globalx_pol = e_localx_pol * np.cos(thetas)
            e_globaly_pol = e_localx_pol * np.sin(thetas)

            ex_sum = np.sum(e_globalx_pol)
            ey_sum = np.sum(e_globaly_pol)

            r_x = calculate_r(freq, ex_sum, area)
            r_y = calculate_r(freq, ey_sum, area)
            # R, T = calculate_R_T(r_x)
            results_x.append((freq, r_x, 1 + r_x))
            results_y.append((freq, r_y, 1 + r_y))
        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)

    return sorted(results_x), sorted(results_y)  # Sort by frequency

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <pols_folder> <area>", file=sys.stderr)
        sys.exit(1)
    
    try:
        folder_path = sys.argv[1]
        area = float(sys.argv[2])
        
        # Process all files and get results
        results_x, results_y = process_pols_folder(folder_path, area)

        if not results_x or not results_y:
            print("No .pols files found or all files failed to process", file=sys.stderr)
            sys.exit(1)
        
        # Write results to CSV
        output_file = os.path.join(folder_path, 'reflection_transmission_complex.csv')
        with open(output_file, 'w') as f:
            f.write("frequency,r_x,t_x,r_y,t_y\n")
            for (freq, r_x, t_x), (_, r_y, t_y) in zip(results_x, results_y):
                f.write(f"{freq:.6e},{r_x:.6e},{t_x:.6e},{r_y:.6e},{t_y:.6e}\n")
                print(f"[{freq:.6e},{r_x:.6e},{t_x:.6e},{r_y:.6e},{t_y:.6e}],")

        print(f"Results written to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()