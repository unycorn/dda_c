#!/usr/bin/env python3
"""
Simple test: two x-oriented dipoles, compute total power
"""

import numpy as np
from scipy.constants import c, mu_0, epsilon_0, pi
from compute_power import compute_full_sphere_power

def test_two_dipoles():
    # Simple parameters
    frequency = 300e12  # 300 THz
    wavelength = c / frequency
    dipole_moment = 1.0e-31  # 1 C⋅m
    separation = wavelength  # 1 wavelength apart
    
    # Two dipoles at (0,0,0) and (λ,0,0)
    positions_array = np.array([
        [0.0, 0.0, 0.0],
        [100e-9, 0.0, 0.0]
    ])
    
    # Both x-oriented with same moment
    pol_array = np.array([
        [complex(dipole_moment, 0.0), complex(0.0, 0.0)],
        [complex(-dipole_moment, 0.0), complex(0.0, 0.0)]
    ])
    
    # Compute power
    sample_R = 1000 * wavelength
    total_power = compute_full_sphere_power(
        positions_array, pol_array, frequency, sample_R, 500, 1000
    )
    
    print(f"Two dipoles total power: {total_power:.3e} W")

if __name__ == "__main__":
    test_two_dipoles()