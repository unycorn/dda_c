import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, mu_0, epsilon_0, pi
import read_polarizations
import dipole
# from numba import jit

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

    width = args.width
    wover2 = width/2
    z = -args.distance
    sample_N = args.density

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

    sample_locations = []
    for x in np.linspace(-wover2, wover2, sample_N):
        for y in np.linspace(-wover2, wover2, sample_N):
            sample_locations.append([x,y,z])

    # Process each frequency
    for d in data_pairs:
        P0 = 0
        Pt = 0

        freq, file, N, polarizations = d
        
        for r_s in sample_locations:
            Ex, By = gaussian_beam_downward(r_s[0], r_s[1], r_s[2], 5e-6, freq)
            P0 += Ex * By / mu_0

            Ey = Bx = 0
            for i in range(N):
                r_p = positions[i]
                px, mz = polarizations[i]
                
                E, H = dipole.calculate_both_dipole_fields(r_p, r_s, [px, 0, 0, 0, 0, mz], 2*pi*freq)
                Ex = Ex + E[0]
                Ey = Ey + E[1]
                Bx = Bx + mu_0 * H[0]
                By = By + mu_0 * H[1]

            Pt += (Ex * By - Ey * Bx) / mu_0
        print(f"{freq*1e-12:.0f} THz", P0, Pt)
            


if __name__ == "__main__":
    main()