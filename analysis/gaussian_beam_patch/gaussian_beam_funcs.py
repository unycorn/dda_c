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

@jit(nopython=True)
def gaussian_beam_jit(x, y, z, w0, freq):
    """Gaussian beam of radius w0 and propagating along the positive x-direction - JIT optimized"""
    wvl = c / freq
    k = 2 * np.pi / wvl

    w0 = 10e-6
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
def calculate_power_at_samples(sample_locations, positions, polarizations, freq, A):
    """Calculate power for all sample locations - JIT optimized"""
    P0 = 0.0
    Pt = 0.0
    N = len(positions)
    
    for i in range(len(sample_locations)):
        r_s = sample_locations[i]
        
        # Calculate incident beam
        Ex, By = gaussian_beam_downward_jit(r_s[0], r_s[1], r_s[2], 5e-6, freq)
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

def gaussian_beam(x, y, z, w0, freq):
    """Gaussian beam of radius w0 and propagating along the positive x-direction"""

    wvl = c / freq
    k = 2 * np.pi / wvl
    omega = c * k

    w0 = 10e-6
    f = 1 / (k * w0)
    l = k * w0**2

    eikz = np.exp(1j * k * z)
    izlfactor = (1 + 1j*z/l)

    r = np.sqrt(x**2 + y**2)

    Ex = eikz/izlfactor * np.exp(-((r)**2) / (2 * (w0)**2 * izlfactor))
    By = Ex / c
    return Ex, By

def gaussian_beam_downward(x,y,z,w0,freq):
    """Gives the z-mirrored beam by making the transformations z -> -z and B -> -B"""
    Ex, By = gaussian_beam(x, y, -z, w0, freq)
    return Ex, -By

z = -0.25e-6
print(gaussian_beam_downward(0, 0, z, 5e-6, 300e12))

def main():
    parser = argparse.ArgumentParser(description='Create animated Voronoi diagram of polarization data')
    parser.add_argument('csv_file', help='CSV file with x,y,z,theta columns')
    parser.add_argument('distance', help='sampling distance')
    parser.add_argument('width', help='width of sampling region')
    parser.add_argument('density', help='number of sampling sites by length')
    args = parser.parse_args()

    width = float(args.width)
    wover2 = width/2
    z = -float(args.distance)
    sample_N = int(args.density)

    A = width**2 / sample_N**2

    # Read position data
    df = pd.read_csv(args.csv_file)
    positions = df[['x', 'y', 'z']].values
    thetas = df['theta'].values

    # Get the pols folder path by removing .csv from the input file path
    pols_folder = os.path.splitext(args.csv_file)[0]
    if not os.path.isdir(pols_folder):
        raise ValueError(f"Could not find polarization data folder: {pols_folder}")
    
    pols_files = glob.glob(os.path.join(pols_folder, "*.pols"))
    # Read frequencies and full data to sort files
    data_pairs = []
    for file in pols_files:
        N, freq, polarizations = read_polarizations.read_polarizations_binary(file)
        data_pairs.append((freq, file, N, polarizations))
    
    # Sort by frequency
    data_pairs.sort()

    # Convert sample locations to numpy arrays for JIT optimization
    sample_locations = np.zeros((sample_N * sample_N, 3))
    idx = 0
    for x in np.linspace(-wover2, wover2, sample_N):
        for y in np.linspace(-wover2, wover2, sample_N):
            sample_locations[idx] = [x, y, z]
            idx += 1

    freq_list = []
    P0_list = []
    Pt_list = []

    # Process each frequency
    for d in data_pairs:
        freq, file, N, polarizations = d
        
        # Convert data to numpy arrays for JIT optimization
        positions_array = np.array(positions)
        polarizations_array = np.array(polarizations)
        
        # Use JIT-optimized function
        P0, Pt = calculate_power_at_samples(sample_locations, positions_array, polarizations_array, freq, A)
        freq_list.append(freq)
        P0_list.append(P0)
        Pt_list.append(Pt)

        print(f"{freq*1e-12:.0f} THz", P0, Pt)
            
    plt.plot(freq_list, P0_list, "incident power")
    plt.plot(freq_list, Pt_list, "transmitted power")
    plt.savefig("powerplot.png")

if __name__ == "__main__":
    main()