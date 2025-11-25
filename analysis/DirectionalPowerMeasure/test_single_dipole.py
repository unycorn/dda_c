#!/usr/bin/env python3
"""
Test compute_power.py against theoretical single dipole radiation.
Creates a single dipole and compares numerical vs analytical results.
"""

import numpy as np
import pandas as pd
from scipy.constants import c, mu_0, epsilon_0, pi
import os
import tempfile
from compute_power import compute_hemisphere_power, compute_full_sphere_power

# Constants
Z_0 = c * mu_0

def theoretical_dipole_power(p_magnitude, frequency):
    """
    Calculate theoretical power radiated by electric dipole
    P = (μ₀ω⁴|p|²)/(12πc)
    
    Parameters:
    - p_magnitude: Magnitude of dipole moment |p| in C⋅m
    - frequency: Frequency in Hz
    
    Returns:
    - Power in Watts
    """
    omega = 2 * pi * frequency
    power = (mu_0 * omega**4 * p_magnitude**2) / (12 * pi * c)
    return power

def create_single_dipole_csv(dipole_moment, filename):
    """Create a CSV file with a single dipole at origin"""
    data = {
        'x': [0.0],
        'y': [0.0], 
        'z': [0.0],
        'theta': [0.0]  # Angle doesn't matter for this test
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return filename

def test_single_dipole():
    """Test numerical computation against analytical result with simple p = 1 C⋅m"""
    
    # Simple test parameters
    frequency = 300e12  # 300 THz
    dipole_moment = 1.0  # 1 C⋅m - simple unit dipole
    
    # Calculate theoretical power
    theoretical_power = theoretical_dipole_power(dipole_moment, frequency)
    
    # Create single dipole data
    positions_array = np.array([[0.0, 0.0, 0.0]])
    pol_array = np.array([[complex(dipole_moment, 0.0), complex(0.0, 0.0)]])
    
    # Set far-field sampling radius  
    wavelength = c / frequency
    sample_R = 1000 * wavelength
    
    print(f"Theoretical power: {theoretical_power:.6e} W")
    print("\nComparing hemisphere doubling vs full sphere integration:")
    
    # Test different resolutions systematically
    resolutions = [
        (50, 100), 
        (100, 200),
        (200, 400),
        (1000, 2000)
    ]
    
    for theta_res, phi_res in resolutions:
        # Hemisphere method (doubled)
        hemisphere_power = compute_hemisphere_power(
            positions_array, pol_array, frequency, sample_R, theta_res, phi_res
        )
        hemisphere_doubled = 2 * hemisphere_power
        hemisphere_error = abs(hemisphere_doubled - theoretical_power) / theoretical_power * 100
        
        # Full sphere method
        full_sphere_power = compute_full_sphere_power(
            positions_array, pol_array, frequency, sample_R, theta_res, phi_res
        )
        full_sphere_error = abs(full_sphere_power - theoretical_power) / theoretical_power * 100
        
        print(f"  {theta_res:3d}×{phi_res:3d}:")
        print(f"    Hemisphere×2: {hemisphere_doubled:.6e} W ({hemisphere_error:.3f}% error)")
        print(f"    Full sphere:  {full_sphere_power:.6e} W ({full_sphere_error:.3f}% error)")

if __name__ == "__main__":
    test_single_dipole()