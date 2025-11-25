#!/usr/bin/env python3
"""
Compute total power radiated into upper hemisphere using far-field Green's functions.
Uses the same technique as directional_plot.py but integrates over the full hemisphere.
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
from scipy.constants import c, mu_0, epsilon_0, pi
import read_polarizations
from numba import jit
import matplotlib.pyplot as plt

# Constants for JIT functions
MU_0 = 1.25663706127e-6
EPSILON_0 = 8.8541878188e-12
Z_0 = c * mu_0

# Add the same Green's function implementations from directional_plot.py
@jit(nopython=True)
def green_E_E_dipole(r_j, r_k, k, farfield=False):
    """Electric field from electric dipole - returns 3x3 Green's tensor"""
    r = r_j - r_k
    r_len = np.sqrt(np.sum(r**2))
    
    if r_len == 0:
        return np.zeros((3, 3), dtype=np.complex128)
    
    r_hat = r / r_len
    expikr = np.exp(1j * k * r_len)
    prefac = 1.0 / (4 * np.pi * EPSILON_0) * expikr / r_len
    
    term1 = k * k
    if farfield == True:
        term2 = 0
    else:
        term2 = (1.0 - 1j * k * r_len) / (r_len * r_len)
    
    # Outer product r_hat ⊗ r_hat
    dyad = np.outer(r_hat, r_hat)
    
    # Identity matrix
    identity = np.eye(3)
    
    out = np.zeros((3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            out[i, j] = prefac * (term2 * (3.0 * dyad[i, j] - identity[i, j]) + 
                                 term1 * (identity[i, j] - dyad[i, j]))
    
    return out

@jit(nopython=True)
def farfield_E_E_dipole(px, r_source, r_sample, k):
    """Calculate far-field electric field from electric dipole"""
    # Create electric dipole moment vector
    p = np.array([px, 0.0+0.0j, 0.0+0.0j])
    
    # Get far-field Green's tensor
    G_EE = green_E_E_dipole(r_sample, r_source, k, farfield=True)
    
    # Calculate E-field: E = G_EE·p
    E_field = np.zeros(3, dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            E_field[i] += G_EE[i, j] * p[j]
    
    return E_field

@jit(nopython=True)
def compute_full_sphere_power(positions_array, polarizations_array, freq, sample_R, n_theta, n_phi):
    """
    Compute total power radiated into full sphere (all directions)
    
    Parameters:
    - positions_array: Array of dipole positions
    - polarizations_array: Array of dipole polarizations [px, mz] 
    - freq: Frequency in Hz
    - sample_R: Radius for far-field sampling
    - n_theta: Number of theta sampling points (0 to π for full sphere)
    - n_phi: Number of phi sampling points (0 to 2π)
    
    Returns:
    - total_power: Total power radiated into full sphere
    """
    k = 2 * pi * freq / c
    
    # Spherical coordinate sampling for full sphere
    # theta: 0 to π (full polar range)
    # phi: 0 to 2π (full azimuthal range)
    theta_vals = np.linspace(0, pi, n_theta)
    phi_vals = np.linspace(0, 2*pi, n_phi)
    
    total_power = 0.0
    
    for i in range(n_theta):
        theta = theta_vals[i]
        
        for j in range(n_phi):
            phi = phi_vals[j]
            
            # Convert spherical to cartesian coordinates
            # x = R*sin(θ)*cos(φ), y = R*sin(θ)*sin(φ), z = R*cos(θ)
            r_sample = sample_R * np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi), 
                np.cos(theta)
            ])
            
            # Calculate total far-field from all dipoles
            E_farfield = np.zeros(3, dtype=np.complex128)
            
            for n in range(len(positions_array)):
                r_source = positions_array[n]
                px = polarizations_array[n, 0]  # Only electric dipole in x-direction
                
                E_farfield += farfield_E_E_dipole(px, r_source, r_sample, k)
            
            # Calculate power density (Poynting vector magnitude)
            # S = (1/2) * (1/Z_0) * |E|^2 for far field
            E_magnitude_sq = np.real(np.dot(np.conj(E_farfield), E_farfield))
            power_density = 0.5 * E_magnitude_sq / Z_0
            
            # Surface element in spherical coordinates: dS = R^2 * sin(θ) * dθ * dφ
            dtheta = pi / (n_theta - 1) if n_theta > 1 else pi
            dphi = 2*pi / (n_phi - 1) if n_phi > 1 else 2*pi
            dS = sample_R**2 * np.sin(theta) * dtheta * dphi
            
            # Add contribution to total power
            total_power += power_density * dS
    
    return total_power

def main():
    parser = argparse.ArgumentParser(description='Compute total power radiated into full sphere')
    parser.add_argument('csv_pattern', help='CSV file pattern (e.g., "data/*.csv" or single file path)')
    parser.add_argument('--incident_power', type=float, required=True, help='Incident power in Watts')
    parser.add_argument('--n_theta', type=int, default=50, help='Number of theta sampling points (default: 50)')
    parser.add_argument('--n_phi', type=int, default=100, help='Number of phi sampling points (default: 100)')
    parser.add_argument('--freq_target', type=float, default=220e12, help='Target frequency in Hz (default: 220 THz)')
    parser.add_argument('--single_freq', action='store_true', help='Only process frequency closest to target (faster)')
    parser.add_argument('--max_freqs', type=int, default=None, help='Maximum number of frequencies to process')
    parser.add_argument('--plot', action='store_true', help='Create plots of power vs frequency')
    parser.add_argument('--save_data', type=str, default=None, help='Save results to CSV file')

    args = parser.parse_args()
    
    # Get list of CSV files matching the pattern
    csv_files = glob.glob(args.csv_pattern)
    if not csv_files:
        raise ValueError(f"No files found matching pattern: {args.csv_pattern}")
    
    csv_files.sort()  # Sort for consistent ordering
    print(f"Found {len(csv_files)} files to process:")
    for file in csv_files:
        print(f"  {file}")

    print(f"\nSampling parameters:")
    print(f"  Theta points: {args.n_theta} (0 to π/2)")
    print(f"  Phi points: {args.n_phi} (0 to 2π)")
    print(f"  Target frequency: {args.freq_target*1e-12:.1f} THz")
    print(f"  Incident power: {args.incident_power:.6e} W")

    # Store results for all files
    all_results = {}
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file}")
        
        # Read position data
        df = pd.read_csv(csv_file)
        positions = df[['x', 'y', 'z']].values
        
        # Get the pols folder path by removing .csv from the input file path
        pols_folder = os.path.splitext(csv_file)[0]
        if not os.path.isdir(pols_folder):
            print(f"Warning: Could not find polarization data folder: {pols_folder}")
            continue
        
        pols_files = glob.glob(os.path.join(pols_folder, "*.pols"))
        if not pols_files:
            print(f"Warning: No .pols files found in {pols_folder}")
            continue
            
        # Read frequencies and full data to sort files
        data_pairs = []
        for file in pols_files:
            try:
                # Try new format with absorption
                N, freq, polarizations, absorption = read_polarizations.read_polarizations_binary(file)
            except:
                # Fall back to old format
                N, freq, polarizations = read_polarizations.read_polarizations_binary(file)
                absorption = None
            data_pairs.append((freq, file, N, polarizations, absorption))
        
        # Sort by frequency
        data_pairs.sort()

        # Convert position data to numpy arrays (same for all frequencies)
        positions_array = np.array(positions)
        
        # Determine which frequencies to process
        if args.single_freq:
            # Find frequency closest to target
            freq_diffs = [abs(freq - args.freq_target) for freq, _, _, _, _ in data_pairs]
            closest_idx = np.argmin(freq_diffs)
            data_pairs = [data_pairs[closest_idx]]
            print(f"Processing single frequency: {data_pairs[0][0]*1e-12:.2f} THz (closest to {args.freq_target*1e-12:.1f} THz)")
        elif args.max_freqs is not None:
            original_count = len(data_pairs)
            data_pairs = data_pairs[:args.max_freqs]
            print(f"Processing first {len(data_pairs)} of {original_count} frequencies")
        else:
            print(f"Found {len(data_pairs)} frequencies to process:")
        
        # Process selected frequencies
        file_results = {}
        for freq_idx, (freq, pols_file, N, polarizations, absorption) in enumerate(data_pairs):
            print(f"\n--- Frequency {freq_idx+1}/{len(data_pairs)}: {freq*1e-12:.2f} THz ---")
            
            if absorption is not None:
                print(f"Absorption from .pols file: {absorption:.6e} W")

            # Convert polarization data to numpy arrays for JIT optimization
            polarizations_array = np.array(polarizations)
            
            # Set far-field sampling radius (should be >> wavelength)
            wavelength = c / freq
            sample_R = 1000 * wavelength  # 1000 wavelengths away
            print(f"Wavelength: {wavelength*1e6:.2f} μm")
            print(f"Far-field sampling radius: {sample_R*1e6:.1f} μm")

            # Compute total radiated power into full sphere
            print(f"Computing full sphere power with {args.n_theta}×{args.n_phi} sampling...")
            total_power = compute_full_sphere_power(
                positions_array, 
                polarizations_array, 
                freq, 
                sample_R, 
                args.n_theta, 
                args.n_phi
            )
            
            # Calculate power components
            scattered_power = total_power
            reflected_power = 0.5 * scattered_power
            absorption_power = absorption if absorption is not None else 0.0
            transmitted_power = args.incident_power - reflected_power - absorption_power
            
            print(f"Results:")
            print(f"  Incident power: {args.incident_power:.6e} W")
            print(f"  Scattered power: {scattered_power:.6e} W")
            print(f"  Reflected power: {reflected_power:.6e} W")
            print(f"  Transmitted power: {transmitted_power:.6e} W")
            
            if absorption is not None:
                print(f"  Absorption power: {absorption_power:.6e} W")
            
            # Store results for this frequency
            freq_key = f"{freq*1e-12:.1f}THz"
            file_results[freq_key] = {
                'frequency': freq,
                'incident_power': args.incident_power,
                'scattered_power': scattered_power,
                'reflected_power': reflected_power,
                'transmitted_power': transmitted_power,
                'absorption_power': absorption_power
            }
        
        # Store all results for this file
        file_key = os.path.basename(csv_file)
        all_results[file_key] = file_results

    # Print summary for all files and frequencies
    print(f"\n{'='*120}")
    print("SUMMARY OF ALL FILES AND FREQUENCIES:")
    print(f"{'='*120}")
    print(f"{'File':<15} {'Freq':<8} {'Incident':<12} {'Scattered':<12} {'Reflected':<12} {'Transmitted':<12} {'Absorbed':<12}")
    print(f"{'':15} {'(THz)':<8} {'(W)':<12} {'(W)':<12} {'(W)':<12} {'(W)':<12} {'(W)':<12}")
    print(f"{'-'*120}")
    
    for file_key, freq_results in all_results.items():
        for freq_key, results in freq_results.items():
            freq_str = f"{results['frequency']*1e-12:.1f}"
            inc_str = f"{results['incident_power']:.2e}"
            scat_str = f"{results['scattered_power']:.2e}"
            ref_str = f"{results['reflected_power']:.2e}"
            trans_str = f"{results['transmitted_power']:.2e}"
            abs_str = f"{results['absorption_power']:.2e}"
            print(f"{file_key:<15} {freq_str:<8} {inc_str:<12} {scat_str:<12} {ref_str:<12} {trans_str:<12} {abs_str:<12}")
        print()  # Blank line between files
    
    # Create plots and save data if requested
    if args.plot or args.save_data:
        # Collect all data for plotting/saving
        all_frequencies = []
        all_incident = []
        all_scattered = []
        all_reflected = []
        all_transmitted = []
        all_absorption = []
        file_labels = []
        
        for file_key, freq_results in all_results.items():
            for freq_key, results in freq_results.items():
                all_frequencies.append(results['frequency'] * 1e-12)  # Convert to THz
                all_incident.append(results['incident_power'])
                all_scattered.append(results['scattered_power'])
                all_reflected.append(results['reflected_power'])
                all_transmitted.append(results['transmitted_power'])
                all_absorption.append(results['absorption_power'])
                file_labels.append(file_key)
        
        all_frequencies = np.array(all_frequencies)
        all_incident = np.array(all_incident)
        all_scattered = np.array(all_scattered)
        all_reflected = np.array(all_reflected)
        all_transmitted = np.array(all_transmitted)
        all_absorption = np.array(all_absorption)
        
        # Save data to CSV if requested
        if args.save_data:
            data_df = pd.DataFrame({
                'File': file_labels,
                'Frequency_THz': all_frequencies,
                'Incident_W': all_incident,
                'Scattered_W': all_scattered,
                'Reflected_W': all_reflected,
                'Transmitted_W': all_transmitted,
                'Absorption_W': all_absorption
            })
            data_df.to_csv(args.save_data, index=False)
            print(f"\nData saved to: {args.save_data}")
        
        # Create plots if requested
        if args.plot:
            plt.figure(figsize=(12, 10))
            
            # Plot all power components
            plt.subplot(2, 1, 1)
            plt.plot(all_frequencies, all_scattered, 'b.-', label='Scattered', linewidth=2, markersize=4)
            plt.plot(all_frequencies, all_reflected, 'r.-', label='Reflected', linewidth=2, markersize=4)
            plt.plot(all_frequencies, all_transmitted, 'g.-', label='Transmitted', linewidth=2, markersize=4)
            plt.plot(all_frequencies, all_absorption, 'm.-', label='Absorbed', linewidth=2, markersize=4)
            plt.xlabel('Frequency (THz)')
            plt.ylabel('Power (W)')
            plt.title('Power Components vs Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            # Plot normalized power (relative to incident)
            plt.subplot(2, 1, 2)
            plt.plot(all_frequencies, all_scattered/all_incident, 'b.-', label='Scattered/Incident', linewidth=2, markersize=4)
            plt.plot(all_frequencies, all_reflected/all_incident, 'r.-', label='Reflected/Incident', linewidth=2, markersize=4)
            plt.plot(all_frequencies, all_transmitted/all_incident, 'g.-', label='Transmitted/Incident', linewidth=2, markersize=4)
            plt.plot(all_frequencies, all_absorption/all_incident, 'm.-', label='Absorbed/Incident', linewidth=2, markersize=4)
            plt.xlabel('Frequency (THz)')
            plt.ylabel('Power Ratio (dimensionless)')
            plt.title('Normalized Power Components vs Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"power_spectrum_{len(csv_files)}files.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved as: {plot_filename}")
            plt.show()

if __name__ == "__main__":
    main()