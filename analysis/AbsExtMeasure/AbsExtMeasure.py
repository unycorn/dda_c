import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, mu_0, epsilon_0, pi
import read_polarizations
import dipole
from numba import jit

# Constants for JIT functions
MU_0 = 1.25663706127e-6
EPSILON_0 = 8.8541878188e-12
Z_0 = c * mu_0

@jit(nopython=True)
def gaussian_beam_jit(x, y, z, w0, freq):
    """Gaussian beam of radius w0 and propagating along the positive x-direction - JIT optimized"""
    wvl = c / freq
    k = 2 * np.pi / wvl

    f = 1 / (k * w0)
    l = k * w0**2

    eikz = np.exp(1j * k * z)
    izlfactor = (1 + 1j*z/l)

    r = np.sqrt(x**2 + y**2)

    Ex = eikz/izlfactor * np.exp(-((r)**2) / (2 * (w0)**2 * izlfactor))
    By = Ex / c
    return Ex, By

@jit(nopython=True)
def gaussian_beam_downward_jit(x, y, z, w0, freq):
    """Gives the z-mirrored beam by making the transformations z -> -z and B -> -B - JIT optimized"""
    Ex, By = gaussian_beam_jit(x, y, -z, w0, freq)
    return Ex, -By

@jit(nopython=True)
def calculate_dipole_fields_jit(r_dipole, r_obs, px, mz, omega):
    """JIT-optimized dipole field calculation for both electric and magnetic dipoles"""
    # Use global constants
    c = 1 / np.sqrt(EPSILON_0 * MU_0)
    k = omega / c
    
    # Distance vector and magnitude
    r_vec = np.array([r_obs[0] - r_dipole[0], r_obs[1] - r_dipole[1], r_obs[2] - r_dipole[2]])
    r_mag = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)
    
    if r_mag == 0:
        return np.array([0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j])
    
    # Unit vector
    r_hat = r_vec / r_mag
    
    # Common exponential factor
    exp_ikr = np.exp(1j * k * r_mag)
    
    # Electric dipole contribution (px only, py=pz=0)
    p = np.array([px, 0.0+0.0j, 0.0+0.0j])
    
    # Electric field from electric dipole
    # E_p(r) = { (1/ε₀)(1/r - ik)(3r̂(r̂·p) - p)/r² + (k²/ε₀)(r̂×(p×r̂))/r } e^(ikr)/(4π)
    r_dot_p = r_hat[0]*p[0] + r_hat[1]*p[1] + r_hat[2]*p[2]
    
    term1_coeff = (1 / (EPSILON_0 * r_mag**3)) - (1j * k / (EPSILON_0 * r_mag**2))
    term1_vec = np.array([3*r_hat[0]*r_dot_p - p[0], 3*r_hat[1]*r_dot_p - p[1], 3*r_hat[2]*r_dot_p - p[2]])
    
    term2_coeff = (k**2) / (EPSILON_0 * r_mag)
    term2_vec = np.array([p[0] - r_hat[0]*r_dot_p, p[1] - r_hat[1]*r_dot_p, p[2] - r_hat[2]*r_dot_p])
    
    E_electric = (term1_coeff * term1_vec + term2_coeff * term2_vec) * (exp_ikr / (4 * np.pi))
    
    # Magnetic field from electric dipole
    # H_p(r) = -iω(1/r - ik)(p×r̂) e^(ikr)/(4πr)
    coeff_H = -1j * omega * (1/r_mag - 1j*k) / (4 * np.pi * r_mag)
    # p×r̂ = [py*rz - pz*ry, pz*rx - px*rz, px*ry - py*rx]
    p_cross_r = np.array([p[1]*r_hat[2] - p[2]*r_hat[1], 
                          p[2]*r_hat[0] - p[0]*r_hat[2], 
                          p[0]*r_hat[1] - p[1]*r_hat[0]])
    H_electric = coeff_H * p_cross_r * exp_ikr
    
    # Magnetic dipole contribution (mz only, mx=my=0)
    m = np.array([0.0+0.0j, 0.0+0.0j, mz])
    
    # Magnetic field from magnetic dipole
    # H_m(r) = { (1/r - ik)(3r̂(r̂·m) - m)/r² + k²(r̂×(m×r̂))/r } e^(ikr)/(4π)
    r_dot_m = r_hat[0]*m[0] + r_hat[1]*m[1] + r_hat[2]*m[2]
    
    term1_coeff_m = (1 / r_mag**3) - (1j * k / r_mag**2)
    term1_vec_m = np.array([3*r_hat[0]*r_dot_m - m[0], 3*r_hat[1]*r_dot_m - m[1], 3*r_hat[2]*r_dot_m - m[2]])
    
    term2_coeff_m = (k**2) / r_mag
    term2_vec_m = np.array([m[0] - r_hat[0]*r_dot_m, m[1] - r_hat[1]*r_dot_m, m[2] - r_hat[2]*r_dot_m])
    
    H_magnetic = (term1_coeff_m * term1_vec_m + term2_coeff_m * term2_vec_m) * (exp_ikr / (4 * np.pi))
    
    # Electric field from magnetic dipole
    # E_m(r) = iωμ₀(1/r - ik)(m×r̂) e^(ikr)/(4πr)
    coeff_E_m = 1j * omega * MU_0 * (1/r_mag - 1j*k) / (4 * np.pi * r_mag)
    # m×r̂ = [my*rz - mz*ry, mz*rx - mx*rz, mx*ry - my*rx]
    m_cross_r = np.array([m[1]*r_hat[2] - m[2]*r_hat[1], 
                          m[2]*r_hat[0] - m[0]*r_hat[2], 
                          m[0]*r_hat[1] - m[1]*r_hat[0]])
    E_magnetic = coeff_E_m * m_cross_r * exp_ikr
    
    # Total fields
    E_total = E_electric + E_magnetic
    H_total = H_electric + H_magnetic
    
    return np.array([E_total[0], E_total[1], E_total[2], H_total[0], H_total[1], H_total[2]])

@jit(nopython=True)
def calculate_radiated_field_jit(sample_location, positions, polarizations, freq):
    """Calculate power for all sample locations - JIT optimized"""
    P0 = 0.0
    Pt = 0.0
    N = len(positions)
    
    r_s = sample_location
    
    # # Calculate incident beam
    # Ex, By = gaussian_beam_downward_jit(r_s[0], r_s[1], r_s[2], 5e-6, freq)
    # P0 += 0.5 * np.real((np.conj(Ex) * By / MU_0) * (-A))
    
    Ex = 0
    By = 0
    
    # Initialize total fields with incident beam
    Ex_total = Ex
    Ey_total = 0.0 + 0.0j
    Ez_total = 0.0 + 0.0j

    Bx_total = 0.0 + 0.0j
    By_total = By
    Bz_total = 0.0 + 0.0j
    
    # Add dipole contributions
    for j in range(N):
        r_p = positions[j]
        px, mz = polarizations[j]

        # If we are closer than 0.1 nm we just skip it
        if np.max(np.abs(r_p - r_s)) > 1e-10:
        
            # Calculate dipole fields
            EH = calculate_dipole_fields_jit(r_p, r_s, px, mz, 2*np.pi*freq)
            Ex_total += EH[0]
            Ey_total += EH[1]
            Ez_total += EH[2]
            Bx_total += MU_0 * EH[3]
            By_total += MU_0 * EH[4]
            Bz_total += MU_0 * EH[5]
    
    return np.array([Ex_total, Ey_total, Ez_total, Bx_total, By_total, Bz_total])

@jit(nopython=True)
def calculate_absorption_extinction_jit(positions_array, polarizations_array, beam_waist, freq):
    """Calculate absorbed and extinguished power for all dipoles - JIT optimized"""
    N = len(positions_array)
    absorbed_power = 0.0
    extinguished_power = 0.0
    
    incident_power_estimate = 0.0

    for r1_i in range(N):
        sample_location = positions_array[r1_i]
        px, mz = polarizations_array[r1_i]
        p_vec = np.array([px, 0.0 + 0.0j, 0.0 + 0.0j])
        
        # Contribution from each dipole
        EB_loc = calculate_radiated_field_jit(sample_location, positions_array, polarizations_array, freq)
        # EB_loc = np.array([0,0,0,0,0,0], dtype=np.complex128)
        
        # Contribution from incident beam
        Einc_x, Binc_y = gaussian_beam_downward_jit(sample_location[0], sample_location[1], sample_location[2], beam_waist, freq)
        incident_power_estimate += -0.5 * np.real(Einc_x * np.conj(Binc_y/mu_0)) * (200e-9)**2
        EB_loc[0] += Einc_x
        EB_loc[4] += Binc_y

        E_loc = EB_loc[:3]
        if r1_i == 0 and freq > 299e12:
            print(round(freq*1e-12), "p estimate", E_loc[0] * (7.1436e-32 + 1.18792e-31j), px)
        E_inc = np.array([Einc_x, 0.0 + 0.0j, 0.0 + 0.0j])

        # extinguished_power += np.pi * freq * np.imag(np.conj(E_inc[0]) * p_vec[0] + np.conj(E_inc[1]) * p_vec[1] + np.conj(E_inc[2]) * p_vec[2])
        # print(px, Einc_x)
        extinguished_power += np.pi * freq * np.imag(np.conj(Einc_x) * px)
        absorbed_power += np.pi * freq * np.imag(np.conj(E_loc[0]) * p_vec[0] + np.conj(E_loc[1]) * p_vec[1] + np.conj(E_loc[2]) * p_vec[2])
    print(freq, incident_power_estimate, extinguished_power)
    return absorbed_power, extinguished_power

@jit(nopython=True)
def calculate_power_at_samples(sample_locations, positions, polarizations, beam_waist, freq, A):
    """Calculate power for all sample locations - JIT optimized"""
    P0 = 0.0
    Pt = 0.0
    N = len(positions)
    
    for i in range(len(sample_locations)):
        r_s = sample_locations[i]
        
        # Calculate incident beam
        Ex, By = gaussian_beam_downward_jit(r_s[0], r_s[1], r_s[2], beam_waist, freq)
        P0 += 0.5 * np.real((np.conj(Ex) * By / MU_0) * (-A))
        
        # Initialize total fields with incident beam
        Ex_total = Ex
        Ey_total = 0.0 + 0.0j
        Bx_total = 0.0 + 0.0j
        By_total = By
        
        # Add dipole contributions
        for j in range(N):
            r_p = positions[j]
            px, mz = polarizations[j]
            
            # Calculate dipole fields
            EH = calculate_dipole_fields_jit(r_p, r_s, px, mz, 2*np.pi*freq)
            Ex_total += EH[0]
            Ey_total += EH[1]
            Bx_total += MU_0 * EH[3]
            By_total += MU_0 * EH[4]
        
        Pt += 0.5 * np.real(((np.conj(Ex_total) * By_total - np.conj(Ey_total) * Bx_total) / MU_0) * (-A))
    
    return P0, Pt

def main():
    parser = argparse.ArgumentParser(description='Calculate absorption, transmission, and reflection for multiple CSV files')
    parser.add_argument('csv_pattern', help='CSV file pattern (e.g., "data/*.csv" or single file path)')

    args = parser.parse_args()
    
    # Get list of CSV files matching the pattern
    csv_files = glob.glob(args.csv_pattern)
    if not csv_files:
        raise ValueError(f"No files found matching pattern: {args.csv_pattern}")
    
    csv_files.sort()  # Sort for consistent ordering
    print(f"Found {len(csv_files)} files to process:")
    for file in csv_files:
        print(f"  {file}")

    beam_waist = 5e-6
    incident_power = pi * beam_waist**2 / ( 2 * Z_0 ) # With a center amplitude of 1 V/m
    print("incident power", incident_power)

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
            N, freq, polarizations = read_polarizations.read_polarizations_binary(file)
            data_pairs.append((freq, file, N, polarizations))
        
        # Sort by frequency
        data_pairs.sort()

        freq_list = []
        absorbed_power_list = []
        extinguished_power_list = []

        # Process each frequency
        for d in data_pairs:
            freq, file, N, polarizations = d
            
            freq_list.append(freq)

            # Convert data to numpy arrays for JIT optimization
            positions_array = np.array(positions)
            polarizations_array = np.array(polarizations)
            
            # Calculate absorbed and extinguished power using JIT-compiled function
            absorbed_power, extinguished_power = calculate_absorption_extinction_jit(positions_array, polarizations_array, beam_waist, freq)
            
            absorbed_power_list.append(absorbed_power)
            extinguished_power_list.append(extinguished_power)

        print(f"  Processed {len(freq_list)} frequencies")
        
        # Save individual file data
        np.save(os.path.join(pols_folder, "power.npy"), np.array([freq_list, incident_power*np.ones_like(freq_list), extinguished_power_list, absorbed_power_list]))
        print(f"  Saved data to {os.path.join(pols_folder, 'power.npy')}")

        # Convert to numpy arrays and calculate A, T, R
        absorbed_power_list = np.array(absorbed_power_list)
        extinguished_power_list = np.array(extinguished_power_list)
        incident_power_list = incident_power * np.ones_like(freq_list)

        A = absorbed_power_list / incident_power_list
        T = 1 - (extinguished_power_list/incident_power_list)
        R = 1 - A - T
        
        # Store results for plotting - use parent directory name for unique labels
        parent_dir = os.path.basename(os.path.dirname(csv_file))
        file_basename = os.path.basename(csv_file).replace('.csv', '')
        file_label = f"{parent_dir}_{file_basename}" if parent_dir else file_basename
        
        # If still not unique, use more of the path
        if file_label in all_results:
            file_label = csv_file.replace('/', '_').replace('.csv', '')
        
        all_results[file_label] = {
            'freq_list': np.array(freq_list),
            'A': A,
            'T': T,
            'R': R
        }
        print(f"  Added {file_label} to results with {len(freq_list)} frequency points")
    
    # Debug: Print what we have for plotting
    print(f"\nPreparing to plot {len(all_results)} datasets:")
    for label, data in all_results.items():
        print(f"  {label}: {len(data['freq_list'])} frequencies, T range: {data['T'].min():.6f} to {data['T'].max():.6f}")
    
    # Create combined plots
    plt.figure(figsize=(12, 8))
    
    # Plot transmission for all files
    plt.subplot(2, 2, 1)
    plot_count = 0
    for label, data in all_results.items():
        plt.plot(np.array(data['freq_list'])*1e-12, data['T'], label=f'{label} - T')
        plot_count += 1
        print(f"    Plotted transmission for {label}")
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Transmission')
    plt.title('Transmission vs Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    print(f"  Transmission subplot: plotted {plot_count} curves")
    
    # Plot absorption for all files
    plt.subplot(2, 2, 2)
    for label, data in all_results.items():
        plt.plot(np.array(data['freq_list'])*1e-12, data['A'], label=f'{label} - A')
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Absorption')
    plt.title('Absorption vs Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot reflection for all files
    plt.subplot(2, 2, 3)
    for label, data in all_results.items():
        plt.plot(np.array(data['freq_list'])*1e-12, data['R'], label=f'{label} - R')
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Reflection')
    plt.title('Reflection vs Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot all three for comparison (first file only, or you can modify this)
    plt.subplot(2, 2, 4)
    if all_results:
        first_label = next(iter(all_results))
        first_data = all_results[first_label]
        plt.plot(np.array(first_data['freq_list'])*1e-12, first_data['A'], 'k-', label='Absorption')
        plt.plot(np.array(first_data['freq_list'])*1e-12, first_data['T'], 'b-', label='Transmission')
        plt.plot(np.array(first_data['freq_list'])*1e-12, first_data['R'], 'r-', label='Reflection')
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Coefficient')
        plt.title(f'A/T/R - {first_label}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("combined_ATR_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create individual transmission plot as before
    plt.figure(figsize=(10, 6))
    plot_count = 0
    for label, data in all_results.items():
        plt.plot(np.array(data['freq_list'])*1e-12, data['T'], label=f'{label}')
        plot_count += 1
        print(f"    Plotted transmission comparison for {label}")
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Transmission')
    plt.title('Transmission Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("Tplot_combined.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Transmission comparison plot: plotted {plot_count} curves")
    
    print(f"\nProcessing complete. Generated plots:")
    print("  - combined_ATR_plot.png (4-panel comparison)")
    print("  - Tplot_combined.png (transmission comparison)")

if __name__ == "__main__":
    main()