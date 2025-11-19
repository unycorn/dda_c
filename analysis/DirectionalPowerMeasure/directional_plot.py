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
def calculate_radiated_field_jit(sample_location, positions, polarizations, freq):
    N = len(positions)
    
    r_s = sample_location
    
    # Initialize total fields with incident beam
    Ex_total = 0.0 + 0.0j
    Ey_total = 0.0 + 0.0j
    Ez_total = 0.0 + 0.0j

    Bx_total = 0.0 + 0.0j
    By_total = 0.0 + 0.0j
    Bz_total = 0.0 + 0.0j
    
    # Add dipole contributions
    for j in range(N):
        r_p = positions[j]
        px, mz = polarizations[j]

        # If we are closer than 0.1 nm we just skip it
        if np.max(np.abs(r_p - r_s)) > 1e-10:
        
            # Calculate dipole fields using correct Green's functions
            EH = calculate_dipole_fields_correct(r_p, r_s, px, mz, 2*np.pi*freq)
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
            Exloc = EB_loc[0]
            Hzloc = EB_loc[5]/mu_0
            pest = (7.1436e-32 + 1.18792e-31j) * Exloc + (-4.56251e-30 + 2.55184e-30j) * Hzloc
            print(round(freq*1e-12), "p estimate", pest , px)
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
            
            # Calculate dipole fields using correct Green's functions
            EH = calculate_dipole_fields_correct(r_p, r_s, px, mz, 2*np.pi*freq)
            Ex_total += EH[0]
            Ey_total += EH[1]
            Bx_total += MU_0 * EH[3]
            By_total += MU_0 * EH[4]
        
        Pt += 0.5 * np.real(((np.conj(Ex_total) * By_total - np.conj(Ey_total) * Bx_total) / MU_0) * (-A))
    
    return P0, Pt



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

            sample_R = 100e-6
            sample_thetas = np.linspace(-pi, pi, 500)
            sample_powers = np.zeros_like(sample_thetas)

            for theta_i, theta in enumerate(sample_thetas):
                Efarfield = 0 + 0j
                r_sample = sample_R * np.array([0, np.cos(theta), np.sin(theta)])
                for r_source, pm_source in zip(positions_array, polarizations_array):
                    px, mz = pm_source
                    Efarfield += farfield_E_E_dipole(px, r_source, r_sample, 2*pi*freq/c)
                sample_powers[theta_i] = np.dot(np.abs(Efarfield),np.abs(Efarfield))/(2*Z_0)

            # Create polar plot of the radiation pattern
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8, 8))
            ax.plot(sample_thetas, sample_powers)
            ax.set_theta_zero_location('N')  # Set 0 degrees at the top
            ax.set_theta_direction(-1)       # Clockwise direction
            ax.set_title(f'Radiation Pattern at {freq*1e-12:.1f} THz', pad=20)
            ax.set_ylabel('Power (W)', labelpad=30)
            ax.grid(True)
            
            # Save the plot
            plt.savefig(f"radiation_pattern_{freq*1e-12:.1f}THz.png", dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    main()