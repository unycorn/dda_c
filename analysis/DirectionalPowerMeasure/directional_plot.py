import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, mu_0, epsilon_0, pi
import read_polarizations
# import dipole
from numba import jit

# Constants for JIT functions
MU_0 = 1.25663706127e-6
EPSILON_0 = 8.8541878188e-12
Z_0 = c * mu_0

# Add these functions here
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
def green_H_E_dipole(r_j, r_k, k):
    """Magnetic field from electric dipole - returns 3x3 Green's tensor"""
    r = r_j - r_k
    r_len = np.sqrt(np.sum(r**2))
    
    if r_len == 0:
        return np.zeros((3, 3), dtype=np.complex128)
    
    expikr = np.exp(1j * k * r_len)
    c = 1 / np.sqrt(EPSILON_0 * MU_0)
    omega = k * c
    prefac = -1j * omega * expikr / (4 * np.pi * r_len * r_len)
    term = (1.0 / r_len - 1j * k)
    
    # Cross product matrix [r×] where [r×]v = r × v
    cross_matrix = np.array([
        [0.0, -r[2], r[1]],
        [r[2], 0.0, -r[0]], 
        [-r[1], r[0], 0.0]
    ], dtype=np.complex128)
    
    out = np.zeros((3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            out[i, j] = -prefac * term * cross_matrix[i, j]
    
    return out

@jit(nopython=True)
def green_E_M_dipole(r_j, r_k, k):
    """Electric field from magnetic dipole - returns 3x3 Green's tensor"""
    r = r_j - r_k
    r_len = np.sqrt(np.sum(r**2))
    
    if r_len == 0:
        return np.zeros((3, 3), dtype=np.complex128)
    
    expikr = np.exp(1j * k * r_len)
    c = 1 / np.sqrt(EPSILON_0 * MU_0)
    omega = k * c
    prefac = 1j * omega * MU_0 * expikr / (4 * np.pi * r_len * r_len)
    term = (1.0 / r_len - 1j * k)
    
    # Cross product matrix [r×] where [r×]v = r × v
    cross_matrix = np.array([
        [0.0, -r[2], r[1]],
        [r[2], 0.0, -r[0]],
        [-r[1], r[0], 0.0]
    ], dtype=np.complex128)
    
    out = np.zeros((3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            out[i, j] = -prefac * term * cross_matrix[i, j]
    
    return out

@jit(nopython=True)
def green_H_M_dipole(r_j, r_k, k):
    """Magnetic field from magnetic dipole - returns 3x3 Green's tensor"""
    r = r_j - r_k
    r_len = np.sqrt(np.sum(r**2))
    
    if r_len == 0:
        return np.zeros((3, 3), dtype=np.complex128)
    
    r_hat = r / r_len
    expikr = np.exp(1j * k * r_len)
    prefac = 1.0 / (4 * np.pi) * expikr / r_len
    
    term1 = k * k
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
def calculate_dipole_fields_correct(r_dipole, r_obs, px, mz, omega):
    """Calculate fields using the correct Green's functions"""
    c = 1 / np.sqrt(EPSILON_0 * MU_0)
    k = omega / c
    
    # Electric dipole moment (px, 0, 0)
    p = np.array([px, 0.0+0.0j, 0.0+0.0j])
    
    # Magnetic dipole moment (0, 0, mz)
    m = np.array([0.0+0.0j, 0.0+0.0j, mz])
    
    # Get Green's function matrices
    G_EE = green_E_E_dipole(r_obs, r_dipole, k)
    G_HE = green_H_E_dipole(r_obs, r_dipole, k)
    G_EM = green_E_M_dipole(r_obs, r_dipole, k)
    G_HM = green_H_M_dipole(r_obs, r_dipole, k)
    
    # Calculate fields: E = G_EE·p + G_EM·m, H = G_HE·p + G_HM·m
    E_total = np.zeros(3, dtype=np.complex128)
    H_total = np.zeros(3, dtype=np.complex128)
    
    for i in range(3):
        for j in range(3):
            E_total[i] += G_EE[i, j] * p[j] + G_EM[i, j] * m[j]
            H_total[i] += G_HE[i, j] * p[j] + G_HM[i, j] * m[j]
    
    return np.array([E_total[0], E_total[1], E_total[2], H_total[0], H_total[1], H_total[2]])

def main():
    parser = argparse.ArgumentParser(description='Calculate absorption, transmission, and reflection for multiple CSV files')
    parser.add_argument('csv_pattern', help='CSV file pattern (e.g., "data/*.csv" or single file path)')
    parser.add_argument('--linear', action='store_true', help='Plot linear power values instead of logarithmic (dB) scale')

    args = parser.parse_args()
    
    # Get list of CSV files matching the pattern
    csv_files = glob.glob(args.csv_pattern)
    if not csv_files:
        raise ValueError(f"No files found matching pattern: {args.csv_pattern}")
    
    csv_files.sort()  # Sort for consistent ordering
    print(f"Found {len(csv_files)} files to process:")
    for file in csv_files:
        print(f"  {file}")

    # beam_waist = 5e-6
    # incident_power = pi * beam_waist**2 / ( 2 * Z_0 ) # With a center amplitude of 1 V/m
    # print("incident power", incident_power)

    # Store results for all files
    all_results = {}
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file}")
        
        # Read position data
        df = pd.read_csv(csv_file)
        positions = df[['x', 'y', 'z']].values
        thetas = df['theta'].values
        
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
            N, freq, polarizations, absorption = read_polarizations.read_polarizations_binary(file)
            data_pairs.append((freq, file, N, polarizations))
        
        # Sort by frequency
        data_pairs.sort()

        # Find frequency closest to 300 THz
        target_freq = 300e12 
        freq_diffs = [abs(freq - target_freq) for freq, _, _, _ in data_pairs]
        closest_idx = np.argmin(freq_diffs)
        selected_freq, selected_file, selected_N, selected_polarizations = data_pairs[closest_idx]
        print(f"Selected frequency: {selected_freq*1e-12:.2f} THz (closest to 300 THz)")

        freq_list = []
        absorbed_power_list = []
        extinguished_power_list = []

        # Create three subplots for XZ, YZ, and XY planes
        fig, (ax_xz, ax_yz, ax_xy) = plt.subplots(1, 3, subplot_kw=dict(projection='polar'), figsize=(24, 8))
        
        # Set up the axes properties (unit circle style)
        for ax in [ax_xz, ax_yz, ax_xy]:
            ax.set_theta_zero_location('E')  # Set 0 degrees on the right
            ax.set_theta_direction(1)        # Counter-clockwise direction
            ax.grid(True)
        
        # Set titles for each plane
        freq_title = f'{selected_freq*1e-12:.1f} THz'
        ax_xz.set_title(f'XZ Plane Radiation Pattern ({freq_title})', pad=20)
        ax_yz.set_title(f'YZ Plane Radiation Pattern ({freq_title})', pad=20)
        ax_xy.set_title(f'XY Plane Radiation Pattern ({freq_title})', pad=20)
        
        # Define cutoff distances to sweep through (in meters), including infinite cutoff
        cutoff_distances = [0.5e-6, 1.0e-6, 1.5e-6, 2.0e-6, 2.5e-6, 3.0e-6, 4.0e-6, np.inf]
        
        # Define colors for different cutoff distances using colormap
        colors = plt.cm.turbo(np.linspace(0, 1, len(cutoff_distances)))

        # Store all power values to calculate shared range
        all_powerDB_values = []
        
        # Set power label based on scale choice
        power_label = 'Power (W)' if args.linear else 'Power (dB)'

        # Convert data to numpy arrays for JIT optimization
        positions_array = np.array(positions)
        polarizations_array = np.array(selected_polarizations)

        sample_R = 10000e-6
        sample_thetas = np.linspace(-pi, pi, 1000)
        
        # Process each cutoff distance
        for cutoff_idx, cutoff_distance in enumerate(cutoff_distances):
            # Handle infinite cutoff case
            if np.isinf(cutoff_distance):
                cutoff_distance_sq = np.inf
            else:
                cutoff_distance_sq = cutoff_distance**2
            
            # Calculate powers for each plane
            sample_powers_xz = np.zeros_like(sample_thetas)  # XZ plane (y=0)
            sample_powers_yz = np.zeros_like(sample_thetas)  # YZ plane (x=0)  
            sample_powers_xy = np.zeros_like(sample_thetas)  # XY plane (z=0)
            
            for theta_i, theta in enumerate(sample_thetas):
                # XZ plane: r_sample = [cos(θ), 0, sin(θ)]
                Efarfield_xz = np.zeros((3), dtype=np.complex128)
                r_sample_xz = sample_R * np.array([np.cos(theta), 0, np.sin(theta)])
                
                # YZ plane: r_sample = [0, cos(θ), sin(θ)]
                Efarfield_yz = np.zeros((3), dtype=np.complex128)
                r_sample_yz = sample_R * np.array([0, np.cos(theta), np.sin(theta)])
                
                # XY plane: r_sample = [cos(θ), sin(θ), 0]
                Efarfield_xy = np.zeros((3), dtype=np.complex128)
                r_sample_xy = sample_R * np.array([np.cos(theta), np.sin(theta), 0])
                
                for r_source, pm_source in zip(positions_array[::1], polarizations_array[::1]):
                    # Skip cutoff check if cutoff_distance_sq is infinite
                    if not np.isinf(cutoff_distance_sq) and np.dot(r_source, r_source) > cutoff_distance_sq:
                        continue
                    # print(r_source)
                    
                    px, mz = pm_source
                    Efarfield_xz += farfield_E_E_dipole(px, r_source, r_sample_xz, 2*pi*selected_freq/c)
                    Efarfield_yz += farfield_E_E_dipole(px, r_source, r_sample_yz, 2*pi*selected_freq/c)
                    Efarfield_xy += farfield_E_E_dipole(px, r_source, r_sample_xy, 2*pi*selected_freq/c)
                
                sample_powers_xz[theta_i] = np.dot(np.abs(Efarfield_xz), np.abs(Efarfield_xz))/(2*Z_0)
                sample_powers_yz[theta_i] = np.dot(np.abs(Efarfield_yz), np.abs(Efarfield_yz))/(2*Z_0)
                sample_powers_xy[theta_i] = np.dot(np.abs(Efarfield_xy), np.abs(Efarfield_xy))/(2*Z_0)

            # Convert to appropriate scale based on user choice
            if args.linear:
                # Use linear power values
                power_xz = sample_powers_xz
                power_yz = sample_powers_yz
                power_xy = sample_powers_xy
            else:
                # Use logarithmic (dB) scale
                power_xz = np.log10(sample_powers_xz)
                power_yz = np.log10(sample_powers_yz)
                power_xy = np.log10(sample_powers_xy)
            
            # Normalize each power array to [0, 1] range for consistent radial scaling
            def normalize_power(power_array):
                min_val = np.min(power_array)
                max_val = np.max(power_array)
                if max_val > min_val:
                    return (power_array - min_val) / (max_val - min_val)
                else:
                    return np.zeros_like(power_array)
            
            power_xz_norm = normalize_power(power_xz)
            power_yz_norm = normalize_power(power_yz)
            power_xy_norm = normalize_power(power_xy)
            
            # Convert angles to degrees for better readability
            sample_thetas_deg = np.degrees(sample_thetas)
            
            # Plot each plane with different colors and labels
            if np.isinf(cutoff_distance):
                label = 'r < ∞'
            else:
                label = f'r < {cutoff_distance*1e6:.1f} μm'
            ax_xz.plot(sample_thetas, power_xz_norm, linewidth=2, color=colors[cutoff_idx], label=label, alpha=0.7)
            ax_yz.plot(sample_thetas, power_yz_norm, linewidth=2, color=colors[cutoff_idx], label=label, alpha=0.7)
            ax_xy.plot(sample_thetas, power_xy_norm, linewidth=2, color=colors[cutoff_idx], label=label, alpha=0.7)

        # Set radial limits for normalized plots (0 to 1)
        # Set radial limits and add legends
        for ax in [ax_xz, ax_yz, ax_xy]:
            ax.set_ylim(0, 1)  # Normalized range from 0 to 1
            ax.set_ylabel('Normalized Power', labelpad=30)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        # Save the combined plot with unique filename based on input CSV (including parent folder)
        csv_basename = os.path.splitext(os.path.basename(csv_file))[0]
        parent_folder = os.path.basename(os.path.dirname(csv_file))
        scale_suffix = "_linear" if args.linear else "_dB"
        freq_suffix = f"_{selected_freq*1e-12:.1f}THz"
        output_filename = f"/Users/dharper/Documents/DDA_C/analysis/DirectionalPowerMeasure/cutoff_sweep_radiation_patterns_{parent_folder}_{csv_basename}{freq_suffix}{scale_suffix}_normalized.pdf"
        
        plt.tight_layout()
        plt.savefig(output_filename, bbox_inches='tight')
        plt.close()
        
        print(f"Saved radiation pattern plot: {output_filename}")

if __name__ == "__main__":
    main()