#!/usr/bin/env python3
"""
Compare R values from ana_sampler with scattering efficiency from hemisphere power calculations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the reflection coefficient data from ana_sampler
reflection_df = pd.read_csv('/Users/dharper/l2_p0_o0_m0/cdm_input_0/reflection_transmission_complex.csv')

# Read the hemisphere power data
power_df = pd.read_csv('power_spectrum_l2_results.csv')

# Extract real and imaginary parts of r_x and calculate |r|^2 (reflection coefficient R)
r_x = reflection_df['r_x'].apply(lambda x: complex(x.replace('j', 'j')))
R_values = np.abs(r_x)**2

# Convert frequencies to THz for consistency
reflection_freq_THz = reflection_df['frequency'] * 1e-12

# Calculate scattering efficiency from power data (already computed in previous script)
from scipy.constants import c, mu_0, epsilon_0

# Constants for normalization
E0 = 1.0  # V/m (unit amplitude)
Z0 = np.sqrt(mu_0 / epsilon_0)  # Impedance of free space
area = (15e-6)**2  # 15 μm × 15 μm in m²
intensity = 0.5 * E0**2 / Z0  # W/m²
incident_power = intensity * area  # W

# Calculate scattering efficiency
# Note: power_df['Scattering_W'] contains FULL SPHERE power from compute_full_sphere_power()
full_sphere_power = power_df['Scattering_W']  # This is already full sphere power
hemisphere_power = full_sphere_power      # Calculate hemisphere power
scattering_efficiency = hemisphere_power / incident_power

# Create comparison plot
plt.figure(figsize=(14, 10))

# Plot 1: R values from ana_sampler
plt.subplot(3, 1, 1)
plt.plot(reflection_freq_THz, R_values, 'r.-', linewidth=2, markersize=4, label='R = |r|² (ana_sampler)')
plt.xlabel('Frequency (THz)')
plt.ylabel('Reflection Coefficient R')
plt.title('Reflection Coefficient R from Analytic Sampler')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Scattering efficiency from hemisphere power
plt.subplot(3, 1, 2)
plt.plot(power_df['Frequency_THz'], scattering_efficiency, 'b.-', linewidth=2, markersize=4, label='Scattering Efficiency (hemisphere)')
plt.xlabel('Frequency (THz)')
plt.ylabel('Scattering Efficiency')
plt.title('Scattering Efficiency from Hemisphere Power Integration')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 3: Direct comparison (interpolate to common frequency grid)
plt.subplot(3, 1, 3)
# Interpolate R values to match power frequency grid
R_interp = np.interp(power_df['Frequency_THz'], reflection_freq_THz, R_values)

plt.plot(power_df['Frequency_THz'], R_interp, 'r.-', linewidth=2, markersize=4, label='R = |r|² (ana_sampler)', alpha=0.7)
plt.plot(power_df['Frequency_THz'], scattering_efficiency, 'b.-', linewidth=2, markersize=4, label='Scattering Efficiency (hemisphere)', alpha=0.7)
plt.xlabel('Frequency (THz)')
plt.ylabel('Coefficient/Efficiency')
plt.title('Direct Comparison: R vs Scattering Efficiency')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('R_vs_scattering_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate correlation and peak values
print("=== COMPARISON ANALYSIS ===")
print(f"Ana_sampler R values:")
print(f"  Peak R: {R_values.max():.6f} at {reflection_freq_THz[R_values.argmax()]:.1f} THz")
print(f"  R range: {R_values.min():.2e} - {R_values.max():.6f}")

print(f"\nHemisphere scattering efficiency:")
print(f"  Peak efficiency: {scattering_efficiency.max():.6f} at {power_df['Frequency_THz'].iloc[scattering_efficiency.argmax()]:.1f} THz")
print(f"  Efficiency range: {scattering_efficiency.min():.2e} - {scattering_efficiency.max():.6f}")

# Compare at common frequencies
correlation = np.corrcoef(R_interp, scattering_efficiency)[0,1]
print(f"\nCorrelation coefficient: {correlation:.4f}")

# Ratio analysis at peak
peak_idx = scattering_efficiency.argmax()
peak_freq = power_df['Frequency_THz'].iloc[peak_idx]
peak_scat_eff = scattering_efficiency.iloc[peak_idx]
peak_R_interp = R_interp[peak_idx]

print(f"\nAt peak frequency ({peak_freq:.1f} THz):")
print(f"  R (ana_sampler): {peak_R_interp:.6f}")
print(f"  Scattering efficiency: {peak_scat_eff:.6f}")
print(f"  Ratio (R/Scattering): {peak_R_interp/peak_scat_eff:.2f}")