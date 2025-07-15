#!/usr/bin/env python3
import sys
import numpy as np
import os
from pathlib import Path

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
    results = []
    
    # Process all .pols files in the folder
    for file_path in sorted(Path(folder_path).glob('*.pols')):
        try:
            N, freq, data = read_polarizations_binary(str(file_path))
            ex_sum = np.sum(data[:, 0])
            r = calculate_r(freq, ex_sum, area)
            R, T = calculate_R_T(r)
            results.append((freq, R, T))
        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)
    
    return sorted(results)  # Sort by frequency

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <pols_folder> <area>", file=sys.stderr)
        sys.exit(1)
    
    try:
        folder_path = sys.argv[1]
        area = float(sys.argv[2])
        
        # Process all files and get results
        results = process_pols_folder(folder_path, area)
        
        if not results:
            print("No .pols files found or all files failed to process", file=sys.stderr)
            sys.exit(1)
        
        # Write results to CSV
        output_file = os.path.join(folder_path, 'reflection_transmission.csv')
        with open(output_file, 'w') as f:
            f.write("frequency,R,T\n")
            for freq, R, T in results:
                f.write(f"{freq:.6e},{R:.6e},{T:.6e}\n")
        
        print(f"Results written to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()