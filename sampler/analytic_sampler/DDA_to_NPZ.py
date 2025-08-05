#!/usr/bin/env python3
"""
DDA to NPZ Converter

This script processes DDA (Discrete Dipole Approximation) simulation data and saves
the spatial coordinates (x, y, z), orientations (theta), polarization data,
and calculated reflection/transmission coefficients for all simulations in a 
comprehensive NPZ file.

This script combines and supersedes the functionality of folder_sample.py by
including both raw polarization data and calculated R/T coefficients.

Usage:
    python DDA_to_NPZ.py <root_folder> <area>

The script expects:
- Root folder containing subfolders with DDA simulation data
- Each subfolder should contain CSV files with dipole coordinates and orientations
- Each subfolder contains .pols files with frequency and polarization data

Output:
- Creates a comprehensive NPZ file with all simulation data including:
  * Spatial coordinates (x, y, z) and orientations (theta) from CSV files
  * Raw polarization data (Ex, Mz components) from .pols files
  * Calculated reflection (R) and transmission (T) coefficients for x and y polarizations
  * Complex reflection coefficients (r) for further analysis
"""

import sys
import numpy as np
import os
from pathlib import Path
import csv

# Physical constants
EPSILON_0 = 8.8541878128e-12  # vacuum permittivity in F/m
SPEED_OF_LIGHT = 299792458.0  # speed of light in m/s

def read_polarizations_binary(filename):
    """
    Read polarization data from binary .pols file
    
    Returns:
    - N: Number of dipoles
    - freq: Frequency value
    - data: Complex polarization data (N, 2) array [Ex, Mz components]
    """
    with open(filename, 'rb') as f:
        # Read N (4-byte integer)
        N = np.fromfile(f, dtype=np.int32, count=1)[0]
        
        # Read frequency (8-byte double)
        freq = np.fromfile(f, dtype=np.float64, count=1)[0]
        
        # Read complex doubles (2 components per point: Ex and Mz)
        data = np.fromfile(f, dtype=np.complex128, count=2*N)
        
        return N, freq, data.reshape(-1, 2)  # reshape to (N, 2) array

def calculate_r(freq, ex_sum, area):
    """
    Calculate reflection coefficient r from electric field sum
    
    Parameters:
    - freq: Frequency in Hz
    - ex_sum: Sum of electric field polarizations
    - area: Area parameter
    
    Returns:
    - r: Complex reflection coefficient
    """
    # Calculate vacuum wavenumber k = 2π/λ = 2πf/c
    k = 2 * np.pi * freq / SPEED_OF_LIGHT
    # Calculate r using the provided formula
    r = 1j * k / (2 * EPSILON_0 * area) * ex_sum
    return r

def calculate_R_T(r):
    """
    Calculate reflection and transmission coefficients from r
    
    Parameters:
    - r: Complex reflection coefficient
    
    Returns:
    - R: Reflection coefficient |r|²
    - T: Transmission coefficient |1 + r|²
    """
    # Calculate reflection coefficient R = |r|²
    R = np.abs(r) ** 2
    # Calculate transmission coefficient T = |1 + r|²
    T = np.abs(1 + r) ** 2
    return R, T

def load_dipole_positions_and_orientations(csv_file_path):
    """
    Load dipole positions (x, y, z) and orientations (theta) from CSV file
    
    Parameters:
    - csv_file_path: Path to the CSV file containing dipole data
    
    Returns:
    - x_positions, y_positions, z_positions, theta_orientations as numpy arrays
    """
    x_positions = []
    y_positions = []
    z_positions = []
    theta_orientations = []
    
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            x_positions.append(float(row['x']))
            y_positions.append(float(row['y']))
            z_positions.append(float(row['z']))
            theta_orientations.append(float(row['theta']))
    
    return (np.array(x_positions), np.array(y_positions), 
            np.array(z_positions), np.array(theta_orientations))

def process_simulation_folder(csv_file_path, data_folder_path, area):
    """
    Process a single simulation with CSV file and corresponding data folder
    
    Parameters:
    - csv_file_path: Path to the CSV file with dipole coordinates and orientations
    - data_folder_path: Path to the folder containing .pols files
    - area: Area parameter for R/T calculations
    
    Returns:
    - Dictionary containing all simulation data
    """
    print(f"Processing simulation:")
    print(f"  CSV file: {csv_file_path}")
    print(f"  Data folder: {data_folder_path}")
    
    # Load dipole positions and orientations from CSV file
    try:
        x_positions, y_positions, z_positions, thetas = load_dipole_positions_and_orientations(csv_file_path)
    except Exception as e:
        print(f"Error loading dipole data from {csv_file_path}: {e}")
        return None
    
    # Initialize storage for all frequency data
    frequencies = []
    all_polarizations_ex = []  # Ex component for all frequencies
    all_polarizations_mz = []  # Mz component for all frequencies
    
    # Initialize storage for reflection/transmission data
    reflection_x = []  # R values for x-polarization
    transmission_x = []  # T values for x-polarization  
    reflection_y = []  # R values for y-polarization
    transmission_y = []  # T values for y-polarization
    r_complex_x = []  # Complex reflection coefficient for x
    r_complex_y = []  # Complex reflection coefficient for y
    
    # Process all .pols files in the data folder
    pols_files = sorted(Path(data_folder_path).glob('*.pols'))
    
    if not pols_files:
        print(f"No .pols files found in {data_folder_path}")
        return None
    
    for file_path in pols_files:
        try:
            N, freq, data = read_polarizations_binary(str(file_path))
            
            # Verify that the number of dipoles matches
            if len(thetas) != N:
                print(f"Warning: CSV dipole count ({len(thetas)}) doesn't match .pols dipole count ({N}) in {file_path}")
                # Skip this file if counts don't match
                continue
            
            # Get polarization data
            e_localx_pol = data[:, 0]  # Ex component in local frame
            
            # Transform to global coordinates (same as folder_sample.py)
            e_globalx_pol = e_localx_pol * np.cos(thetas)
            e_globaly_pol = e_localx_pol * np.sin(thetas)
            
            # Calculate sums for reflection/transmission
            ex_sum = np.sum(e_globalx_pol)
            ey_sum = np.sum(e_globaly_pol)
            
            # Calculate reflection coefficients
            r_x = calculate_r(freq, ex_sum, area)
            r_y = calculate_r(freq, ey_sum, area)
            
            # Calculate R and T
            R_x, T_x = calculate_R_T(r_x)
            R_y, T_y = calculate_R_T(r_y)
            
            # Store frequency and polarization data
            frequencies.append(freq)
            all_polarizations_ex.append(data[:, 0])  # Ex component
            all_polarizations_mz.append(data[:, 1])  # Mz component
            
            # Store reflection/transmission data
            reflection_x.append(R_x)
            transmission_x.append(T_x)
            reflection_y.append(R_y)
            transmission_y.append(T_y)
            r_complex_x.append(r_x)
            r_complex_y.append(r_y)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)
            continue
    
    if not frequencies:
        print(f"No valid data found in {data_folder_path}")
        return None
    
    # Convert to numpy arrays
    frequencies = np.array(frequencies)
    all_polarizations_ex = np.array(all_polarizations_ex)  # Shape: (n_frequencies, n_dipoles)
    all_polarizations_mz = np.array(all_polarizations_mz)  # Shape: (n_frequencies, n_dipoles)
    
    # Convert R/T data to numpy arrays
    reflection_x = np.array(reflection_x)
    transmission_x = np.array(transmission_x)
    reflection_y = np.array(reflection_y)
    transmission_y = np.array(transmission_y)
    r_complex_x = np.array(r_complex_x)
    r_complex_y = np.array(r_complex_y)
    
    # Create simulation data dictionary
    simulation_data = {
        'folder_path': data_folder_path,
        'csv_file_path': csv_file_path,
        'frequencies': frequencies,
        'x_positions': x_positions,
        'y_positions': y_positions,
        'z_positions': z_positions,
        'theta_orientations': thetas,
        'polarizations_ex': all_polarizations_ex,
        'polarizations_mz': all_polarizations_mz,
        'reflection_x': reflection_x,
        'transmission_x': transmission_x,
        'reflection_y': reflection_y,
        'transmission_y': transmission_y,
        'r_complex_x': r_complex_x,
        'r_complex_y': r_complex_y,
        'n_frequencies': len(frequencies),
        'n_dipoles': len(thetas)
    }
    
    print(f"  Loaded {len(frequencies)} frequencies, {len(thetas)} dipoles")
    
    return simulation_data

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <root_folder> <area>", file=sys.stderr)
        print("  root_folder: Path to folder containing simulation subfolders")
        print("  area: Area parameter (for consistency with original script)")
        sys.exit(1)
    
    root_folder_path = sys.argv[1]
    area = float(sys.argv[2])
    
    print(f"Processing DDA simulations in: {root_folder_path}")
    print(f"Area parameter: {area}")
    
    # Find all simulation folders
    all_simulation_data = []
    folder_names = []
    
    # Process each subfolder
    for subfolder_path in Path(root_folder_path).glob('*/'):
        subfolder_path_str = str(subfolder_path).rstrip('/')
        
        # Find CSV files in this subfolder
        csv_files = list(Path(subfolder_path_str).glob('*.csv'))
        
        if not csv_files:
            print(f"No CSV files found in {subfolder_path_str}. Skipping subfolder.")
            continue
            
        # Process each CSV file and its corresponding data folder
        for csv_file in csv_files:
            csv_name = csv_file.stem  # filename without extension
            data_folder_path = Path(subfolder_path_str) / csv_name
            
            if not data_folder_path.exists() or not data_folder_path.is_dir():
                print(f"Data folder {data_folder_path} not found for CSV {csv_file.name}. Skipping.")
                continue
                
            print(f"Processing CSV: {csv_file.name} with data folder: {csv_name}")
            
            simulation_data = process_simulation_folder(str(csv_file), str(data_folder_path), area)
            
            if simulation_data is not None:
                all_simulation_data.append(simulation_data)
                folder_names.append(f"{Path(subfolder_path_str).name}_{csv_name}")
            else:
                print(f"Skipped simulation: {csv_file.name}")
    
    if not all_simulation_data:
        print("No valid simulation data found!")
        sys.exit(1)
    
    print(f"\nSuccessfully processed {len(all_simulation_data)} simulation folders")
    
    # Prepare data for NPZ storage
    print("Preparing data for NPZ storage...")
    
    # Create comprehensive documentation string
    documentation = f"""
DDA Simulation Data NPZ File Documentation
==========================================

This file contains comprehensive DDA (Discrete Dipole Approximation) simulation data
including spatial coordinates, orientations, polarization responses, and calculated
optical properties.

Generated by: DDA_to_NPZ.py
Total simulations: {len(all_simulation_data)}
Area parameter: {area}

DATA STRUCTURE:
==============

Global Arrays:
--------------
- folder_names: Array of simulation folder names
- folder_paths: Array of full folder paths to simulation directories
- csv_file_paths: Array of CSV file paths containing dipole coordinate data
- n_simulations: Total number of simulations processed
- area_parameter: Area parameter used for reflection/transmission calculations
- documentation: This explanatory text

Individual Simulation Data (prefix: sim_XXXXX_):
--------------------------------------------
For each simulation XXXXX (00000, 00001, 00002, ...):

Coordinate and Orientation Data:
- sim_XXXXX_x_positions: X coordinates of dipoles (meters) from CSV
- sim_XXXXX_y_positions: Y coordinates of dipoles (meters) from CSV  
- sim_XXXXX_z_positions: Z coordinates of dipoles (meters) from CSV
- sim_XXXXX_theta_orientations: Dipole orientations (radians) from CSV

Frequency and Polarization Data:
- sim_XXXXX_frequencies: Frequency array (Hz) for this simulation
- sim_XXXXX_polarizations_ex: Complex electric field Ex component (local frame)
  Shape: (n_frequencies, n_dipoles)
- sim_XXXXX_polarizations_mz: Complex magnetic field Mz component (local frame)
  Shape: (n_frequencies, n_dipoles)

Optical Properties (calculated from polarization data):
- sim_XXXXX_reflection_x: Reflection coefficient R = |r|² for x-polarization
- sim_XXXXX_transmission_x: Transmission coefficient T = |1+r|² for x-polarization
- sim_XXXXX_reflection_y: Reflection coefficient R = |r|² for y-polarization  
- sim_XXXXX_transmission_y: Transmission coefficient T = |1+r|² for y-polarization
- sim_XXXXX_r_complex_x: Complex reflection coefficient r for x-polarization
- sim_XXXXX_r_complex_y: Complex reflection coefficient r for y-polarization

Metadata:
- sim_XXXXX_n_frequencies: Number of frequency points in this simulation
- sim_XXXXX_n_dipoles: Number of dipoles in this simulation

CALCULATION DETAILS:
===================

Coordinate Transformations:
- Local to global: e_global_x = e_local_x * cos(theta)
                  e_global_y = e_local_x * sin(theta)

Reflection Coefficient:
- r = (i * k / (2 * ε₀ * area)) * Σ(E_polarization)
- where k = 2π * frequency / c (vacuum wavenumber)
- ε₀ = 8.854 × 10⁻¹² F/m (vacuum permittivity)
- c = 2.998 × 10⁸ m/s (speed of light)
This is derived from the normal plane wave component radiated by a
super cell of dipoles with periodicity determined by the area parameter.

Optical Coefficients:
- Reflection: R = |r|²
- Transmission: T = |1 + r|²

USAGE EXAMPLE:
=============
import numpy as np

# Load the data
data = np.load('DDA_simulation_data.npz', allow_pickle=True)

# Print this documentation
print(data['documentation'].item())

# Access simulation 0 data
frequencies = data['sim_00000_frequencies']
x_pos = data['sim_00000_x_positions'] 
reflection_x = data['sim_00000_reflection_x']

# List all simulations
n_sims = data['n_simulations'].item()
folder_names = data['folder_names']
for i in range(n_sims):
    print(f"Simulation {{i:05d}}: {{folder_names[i]}}")
"""

    # Create structured arrays for storage
    npz_data = {
        'documentation': documentation,
        'folder_names': np.array(folder_names, dtype=object),
        'folder_paths': np.array([data['folder_path'] for data in all_simulation_data], dtype=object),
        'csv_file_paths': np.array([data['csv_file_path'] for data in all_simulation_data], dtype=object),
        'n_simulations': len(all_simulation_data),
        'area_parameter': area
    }
    
    # Store individual simulation data
    for i, sim_data in enumerate(all_simulation_data):
        prefix = f'sim_{i:05d}'
        npz_data[f'{prefix}_frequencies'] = sim_data['frequencies']
        npz_data[f'{prefix}_x_positions'] = sim_data['x_positions']
        npz_data[f'{prefix}_y_positions'] = sim_data['y_positions']
        npz_data[f'{prefix}_z_positions'] = sim_data['z_positions']
        npz_data[f'{prefix}_theta_orientations'] = sim_data['theta_orientations']
        npz_data[f'{prefix}_polarizations_ex'] = sim_data['polarizations_ex']
        npz_data[f'{prefix}_polarizations_mz'] = sim_data['polarizations_mz']
        npz_data[f'{prefix}_reflection_x'] = sim_data['reflection_x']
        npz_data[f'{prefix}_transmission_x'] = sim_data['transmission_x']
        npz_data[f'{prefix}_reflection_y'] = sim_data['reflection_y']
        npz_data[f'{prefix}_transmission_y'] = sim_data['transmission_y']
        npz_data[f'{prefix}_r_complex_x'] = sim_data['r_complex_x']
        npz_data[f'{prefix}_r_complex_y'] = sim_data['r_complex_y']
        npz_data[f'{prefix}_n_frequencies'] = sim_data['n_frequencies']
        npz_data[f'{prefix}_n_dipoles'] = sim_data['n_dipoles']
    
    # Save to NPZ file
    output_npz = os.path.join(root_folder_path, 'DDA_simulation_data.npz')
    np.savez_compressed(output_npz, **npz_data)
    
    print(f"\nDDA simulation data saved to: {output_npz}")
    print(f"Documentation embedded in NPZ file - access with data['documentation'].item()")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total simulations: {len(all_simulation_data)}")
    print(f"  Output file: {output_npz}")
    print(f"  File size: {os.path.getsize(output_npz) / (1024*1024):.2f} MB")
    
    # Print details for each simulation
    print(f"\nSimulation details:")
    for i, (name, data) in enumerate(zip(folder_names, all_simulation_data)):
        print(f"  {i+1:2d}. {name}: {data['n_frequencies']} frequencies, {data['n_dipoles']} dipoles")
    
    print(f"\nTo view full documentation later, run:")
    print(f"  import numpy as np")
    print(f"  data = np.load('{output_npz}', allow_pickle=True)")
    print(f"  print(data['documentation'].item())")
    
    print(f"\nQuick data structure summary:")
    print(f"  - documentation: Complete variable descriptions and usage examples")
    print(f"  - folder_names: Array of simulation folder names")
    print(f"  - folder_paths: Array of full folder paths")
    print(f"  - csv_file_paths: Array of CSV file paths with dipole data")
    print(f"  - n_simulations: Total number of simulations")
    print(f"  - area_parameter: Area parameter used")
    print(f"  - sim_XXXXX_*: Individual simulation data with prefix sim_XXXXX")
    print(f"    * frequencies: Frequency array for simulation XXXXX")
    print(f"    * x/y/z_positions: Dipole coordinates from CSV for simulation XXXXX")
    print(f"    * theta_orientations: Dipole orientations from CSV for simulation XXXXX")
    print(f"    * polarizations_ex/mz: Polarization data for simulation XXXXX")
    print(f"    * reflection_x/y: Reflection coefficients R = |r|² for x/y polarizations")
    print(f"    * transmission_x/y: Transmission coefficients T = |1+r|² for x/y polarizations")
    print(f"    * r_complex_x/y: Complex reflection coefficients for x/y polarizations")
    print(f"    * n_frequencies/n_dipoles: Counts for simulation XXXXX")

if __name__ == "__main__":
    main()
