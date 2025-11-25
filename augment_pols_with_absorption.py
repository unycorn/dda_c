#!/usr/bin/env python3
"""
Augment .pols files with absorption calculations.

This script reads existing .pols files that may not have absorption data at the end,
computes the absorption using the same formula as the C++ code, and writes updated
files with the absorption value appended.

The absorption formula used is:
absorbed_power_total = (omega / 2) * imag(sum over all dipoles of (p*, m*) * alpha^(-1)† * (p, m))

Where:
- p, m are the electric and magnetic polarizations from the .pols file
- alpha is the 2x2 polarizability matrix computed from Lorentzian parameters
- omega = 2 * pi * frequency

Author: Generated to match main.cpp implementation
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Physical constants (matching constants.hpp)
EPSILON_0 = 8.854187817e-12
MU_0 = 1.2566370614e-6
Z_0 = 376.730313412
C_LIGHT = 2.99792458e8
PI = np.pi


def lorentz_alpha_params(freq, params_dict):
    """
    Compute polarizability using Lorentzian parameters.
    
    Matches the C++ lorentz_alpha_params function in lorentzian.cpp
    """
    f = freq
    f0 = params_dict['F0']
    gamma = params_dict['gamma']
    A = params_dict['A']
    B = params_dict['B']
    C = params_dict['C']
    form = params_dict['form']
    
    # Complex denominator
    denom = (f0**2 - f**2) - 1j * f * gamma
    
    # Numerator and scale factor depend on the form
    if form == 'STANDARD':
        numerator = A
        scale_factor = EPSILON_0
    elif form == 'IF_NUMERATOR':
        numerator = 1j * f * A
        scale_factor = EPSILON_0 * Z_0
    elif form == 'NEG_IF_NUMERATOR':
        numerator = -1j * f * A
        scale_factor = 1.0 / Z_0
    elif form == 'F2_NUMERATOR':
        numerator = f**2 * A
        scale_factor = 1.0
    else:
        raise ValueError(f"Unknown Lorentzian form: {form}")
    
    # Compute normalized polarizability
    norm_alpha = numerator / denom + B + C * f
    
    return scale_factor * norm_alpha


def generate_polarizability_matrix(freq, params_list):
    """
    Generate 2x2 polarizability matrix for a single dipole at given frequency.
    
    Args:
        freq: Frequency in Hz
        params_list: List of 4 parameter dictionaries for elements [00, 05, 50, 55]
    
    Returns:
        2x2 complex numpy array representing the polarizability matrix
    """
    alpha = np.zeros((2, 2), dtype=complex)
    
    # Generate each matrix element
    positions = [(0,0), (0,1), (1,0), (1,1)]
    for idx, (i, j) in enumerate(positions):
        alpha[i, j] = lorentz_alpha_params(freq, params_list[idx])
    
    return alpha


def read_pols_file_old_format(filename):
    """
    Read a .pols file in the old format (without absorption at the end).
    
    Returns:
        N: Number of dipoles
        freq: Frequency in Hz  
        polarizations: Array of shape (N, 2) with complex polarizations [Ex, Mz]
    """
    with open(filename, 'rb') as f:
        # Read N (4-byte integer)
        N = np.fromfile(f, dtype=np.int32, count=1)[0]
        
        # Read frequency (8-byte double)
        freq = np.fromfile(f, dtype=np.float64, count=1)[0]
        
        # Read complex doubles (2 components per dipole: Ex and Mz)
        data = np.fromfile(f, dtype=np.complex128, count=2*N)
        
        return N, freq, data.reshape(N, 2)


def read_pols_file_with_absorption(filename):
    """
    Read a .pols file in the new format (with absorption at the end).
    
    Returns:
        N: Number of dipoles
        freq: Frequency in Hz  
        polarizations: Array of shape (N, 2) with complex polarizations [Ex, Mz]
        absorption: Absorbed power in Watts
    """
    with open(filename, 'rb') as f:
        # Read N (4-byte integer)
        N = np.fromfile(f, dtype=np.int32, count=1)[0]
        
        # Read frequency (8-byte double)
        freq = np.fromfile(f, dtype=np.float64, count=1)[0]
        
        # Read complex doubles (2 components per dipole: Ex and Mz)
        data = np.fromfile(f, dtype=np.complex128, count=2*N)
        
        # Read absorption value (8-byte double)
        absorption = np.fromfile(f, dtype=np.float64, count=1)[0]
        
        return N, freq, data.reshape(N, 2), absorption


def check_file_has_absorption(filename):
    """
    Check if a .pols file already has absorption data by comparing file size.
    
    Returns:
        True if file has absorption data, False otherwise
    """
    try:
        with open(filename, 'rb') as f:
            # Read N and frequency
            N = np.fromfile(f, dtype=np.int32, count=1)[0]
            freq = np.fromfile(f, dtype=np.float64, count=1)[0]
            
            # Calculate expected file size without absorption
            expected_size_without_absorption = 4 + 8 + 2 * N * 16  # int + double + N * 2 * complex128
            
            # Get actual file size
            f.seek(0, 2)  # Seek to end
            actual_size = f.tell()
            
            # If file is 8 bytes larger, it likely has the absorption double
            return actual_size == expected_size_without_absorption + 8
            
    except Exception:
        return False


def write_pols_file_with_absorption(filename, N, freq, polarizations, absorption):
    """
    Write a .pols file with absorption data at the end.
    
    Args:
        filename: Output filename
        N: Number of dipoles
        freq: Frequency in Hz
        polarizations: Array of shape (N, 2) with complex polarizations [Ex, Mz]
        absorption: Absorbed power in Watts
    """
    with open(filename, 'wb') as f:
        # Write N (4-byte integer)
        np.array([N], dtype=np.int32).tofile(f)
        
        # Write frequency (8-byte double)
        np.array([freq], dtype=np.float64).tofile(f)
        
        # Write polarization data (flatten to 1D array)
        polarizations.flatten().astype(np.complex128).tofile(f)
        
        # Write absorption value (8-byte double)
        np.array([absorption], dtype=np.float64).tofile(f)


def compute_absorption(freq, polarizations, dipole_params_list):
    """
    Compute absorbed power using the same formula as main.cpp.
    
    Args:
        freq: Frequency in Hz
        polarizations: Array of shape (N, 2) with complex polarizations [Ex, Mz]
        dipole_params_list: List of parameter lists, one per dipole, each containing
                           4 parameter dictionaries for the matrix elements
    
    Returns:
        absorbed_power_total: Total absorbed power in Watts
    """
    omega = 2.0 * PI * freq
    N = len(polarizations)
    
    absorbed_power_sum = 0.0 + 0.0j
    
    for j in range(N):
        px = polarizations[j, 0]  # Electric polarization Ex
        mz = polarizations[j, 1]  # Magnetic polarization Mz
        
        # Generate polarizability matrix for this dipole
        alpha = generate_polarizability_matrix(freq, dipole_params_list[j])
        
        # Invert 2x2 alpha matrix using determinant method
        det = alpha[0, 0] * alpha[1, 1] - alpha[0, 1] * alpha[1, 0]
        alpha_inv = np.array([[alpha[1, 1], -alpha[0, 1]],
                              [-alpha[1, 0], alpha[0, 0]]]) / det
        
        # Take Hermitian conjugate (dagger) of alpha_inv
        alpha_inv_dagger = np.conj(alpha_inv).T
        
        # Calculate (px*, mz*) * alpha_inv^† * (px; mz) for dipole j
        polarization_conj = np.array([np.conj(px), np.conj(mz)])
        polarization = np.array([px, mz])
        
        # Matrix multiplication: (px*, mz*) * alpha_inv^†
        temp = np.dot(polarization_conj, alpha_inv_dagger)
        
        # Final multiplication with (px; mz), but need to account for units
        # From C++: Ex_star * px + MU_0 * Hz_star * mz
        Ex_star = temp[0]
        Hz_star = temp[1]
        absorbed_power_complex = Ex_star * px + MU_0 * Hz_star * mz
        
        absorbed_power_sum += absorbed_power_complex
    
    absorbed_power_total = (omega / 2.0) * np.imag(absorbed_power_sum)
    return absorbed_power_total


def load_csv_parameters(csv_file):
    """
    Load dipole parameters from CSV file.
    
    Expected CSV format (matching main.cpp):
    x,y,z,angle,f0_00,gamma_00,A_00,B_00,C_00,f0_05,gamma_05,A_05,B_05,C_05,f0_50,gamma_50,A_50,B_50,C_50,f0_55,gamma_55,A_55,B_55,C_55
    
    Returns:
        List of parameter lists, one per dipole, each containing 4 parameter dictionaries
    """
    # Read CSV as raw values (no column names since main.cpp uses indices)
    df = pd.read_csv(csv_file, header=0)  # Skip header but read all data
    
    dipole_params_list = []
    
    for _, row in df.iterrows():
        # Convert row to list of values (matching main.cpp indices)
        values = row.values.tolist()
        
        # Create parameter dictionaries for each matrix element
        # Indices from main.cpp: positions at 0,1,2; angle at 3; then parameters starting at 4
        params_00 = {
            'A': values[6], 'B': values[7], 'C': values[8],  # A, B, C
            'F0': values[4], 'gamma': values[5],             # F0, gamma  
            'form': 'STANDARD'
        }
        
        params_05 = {
            'A': values[11], 'B': values[12], 'C': values[13],  # A, B, C
            'F0': values[9], 'gamma': values[10],               # F0, gamma
            'form': 'IF_NUMERATOR'
        }
        
        params_50 = {
            'A': values[16], 'B': values[17], 'C': values[18],  # A, B, C
            'F0': values[14], 'gamma': values[15],              # F0, gamma
            'form': 'NEG_IF_NUMERATOR'
        }
        
        params_55 = {
            'A': values[21], 'B': values[22], 'C': values[23],  # A, B, C
            'F0': values[19], 'gamma': values[20],              # F0, gamma
            'form': 'F2_NUMERATOR'
        }
        
        dipole_params_list.append([params_00, params_05, params_50, params_55])
    
    return dipole_params_list


def process_pols_folder(folder_path, csv_file, backup=True, force=False):
    """
    Process all .pols files in a folder, adding absorption data.
    
    Args:
        folder_path: Path to folder containing .pols files
        csv_file: Path to CSV file with dipole parameters
        backup: Whether to create backup copies of original files
        force: Whether to overwrite files that already have absorption data
    """
    # Load dipole parameters
    print(f"Loading dipole parameters from {csv_file}")
    dipole_params_list = load_csv_parameters(csv_file)
    
    # Find all .pols files
    pols_files = glob.glob(os.path.join(folder_path, "*.pols"))
    if not pols_files:
        print(f"No .pols files found in {folder_path}")
        return
    
    print(f"Found {len(pols_files)} .pols files")
    
    processed_count = 0
    skipped_count = 0
    
    for pols_file in sorted(pols_files):
        filename = os.path.basename(pols_file)
        
        # Check if file already has absorption data
        has_absorption = check_file_has_absorption(pols_file)
        
        if has_absorption and not force:
            print(f"Skipping {filename} - already has absorption data (use --force to overwrite)")
            skipped_count += 1
            continue
        
        try:
            # Read the file
            if has_absorption:
                N, freq, polarizations, old_absorption = read_pols_file_with_absorption(pols_file)
                print(f"Processing {filename} (freq={freq:.2e} Hz, old_absorption={old_absorption:.6e})")
            else:
                N, freq, polarizations = read_pols_file_old_format(pols_file)
                print(f"Processing {filename} (freq={freq:.2e} Hz, no previous absorption)")
            
            # Verify dipole count matches CSV
            if len(dipole_params_list) != N:
                print(f"Warning: CSV has {len(dipole_params_list)} dipoles but .pols file has {N}")
                continue
            
            # Compute absorption
            absorption = compute_absorption(freq, polarizations, dipole_params_list)
            print(f"  Computed absorption: {absorption:.6e} W")
            
            # Create backup if requested
            if backup and not has_absorption:
                backup_file = pols_file + ".bak"
                if not os.path.exists(backup_file):
                    import shutil
                    shutil.copy2(pols_file, backup_file)
                    print(f"  Created backup: {backup_file}")
            
            # Write updated file
            write_pols_file_with_absorption(pols_file, N, freq, polarizations, absorption)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"\nSummary: Processed {processed_count} files, skipped {skipped_count} files")


def test_lorentzian_function():
    """
    Test the Lorentzian function against sample values.
    """
    print("Testing Lorentzian function...")
    
    # Test with sample parameters
    freq = 220e12  # Hz, close to typical resonance
    
    # Sample parameter set for testing
    test_params = {
        'A': 3.91e7,
        'B': 6.41e-22,
        'C': 2.96e-36,
        'F0': 219.8e12,
        'gamma': 15.0e12,
        'form': 'STANDARD'
    }
    
    alpha = lorentz_alpha_params(freq, test_params)
    print(f"  Sample α = {alpha:.3e}")
    
    print("Test completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Augment .pols files with absorption calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single folder
  python augment_pols_with_absorption.py folder_name/ input.csv
  
  # Process without creating backups
  python augment_pols_with_absorption.py folder_name/ input.csv --no-backup
  
  # Force overwrite files that already have absorption data
  python augment_pols_with_absorption.py folder_name/ input.csv --force
  
  # Test Lorentzian function
  python augment_pols_with_absorption.py --test
        """
    )
    
    parser.add_argument('pols_folder', nargs='?', help='Folder containing .pols files')
    parser.add_argument('csv_file', nargs='?', help='CSV file with dipole parameters')
    parser.add_argument('--no-backup', action='store_true', 
                        help='Do not create backup copies of original files')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite files that already have absorption data')
    parser.add_argument('--test', action='store_true',
                        help='Run test of Lorentzian function')
    
    args = parser.parse_args()
    
    # Handle test mode
    if args.test:
        test_lorentzian_function()
        return
    
    # Validate required arguments for normal operation
    if not args.pols_folder or not args.csv_file:
        parser.print_help()
        sys.exit(1)
    
    # Validate inputs
    if not os.path.isdir(args.pols_folder):
        print(f"Error: {args.pols_folder} is not a valid directory")
        sys.exit(1)
    
    if not os.path.isfile(args.csv_file):
        print(f"Error: {args.csv_file} is not a valid file")
        sys.exit(1)
    
    # Process the folder
    process_pols_folder(args.pols_folder, args.csv_file, 
                       backup=not args.no_backup, force=args.force)


if __name__ == "__main__":
    main()