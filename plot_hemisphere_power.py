#!/usr/bin/env python3
"""
Plot normalized hemisphere power from the saved CSV data
Normalized by power flux of unit electric field amplitude plane wave over 15 μm × 15 μm surface
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, mu_0, epsilon_0

# Read the saved data
df = pd.read_csv('power_spectrum_l2_results.csv')

# Calculate hemisphere power (half of scattering power)
hemisphere_power = df['Scattering_W'] / 2

# Calculate normalization factor: power flux of unit amplitude plane wave over 15×15 μm surface
# For a plane wave with electric field amplitude E0 = 1 V/m:
# Intensity = (1/2) * (1/Z0) * |E0|^2 = (1/2) * (1/Z0) * 1^2
# Power = Intensity × Area
E0 = 1.0  # V/m (unit amplitude)
Z0 = np.sqrt(mu_0 / epsilon_0)  # Impedance of free space
area = (16.8e-6)**2  # 15 μm × 15 μm in m²

intensity = 0.5 * E0**2 / Z0  # W/m²
incident_power = intensity * area  # W

print(f"Normalization parameters:")
print(f"  Unit electric field amplitude: {E0} V/m")
print(f"  Free space impedance Z0: {Z0:.1f} Ω")
print(f"  Surface area: {area*1e12:.0f} μm²")
print(f"  Incident intensity: {intensity:.6e} W/m²")
print(f"  Incident power: {incident_power:.6e} W")

# Normalize hemisphere power by incident power (scattering efficiency)
scattering_efficiency = hemisphere_power / incident_power

# Create the plot
plt.figure(figsize=(12, 8))
plt.plot(df['Frequency_THz'], scattering_efficiency, 'b.-', linewidth=2, markersize=6)
plt.xlabel('Frequency (THz)')
plt.ylabel('Scattering Efficiency (Hemisphere)')
plt.title('Normalized Hemisphere Scattering Efficiency vs Frequency\n(Normalized by unit amplitude plane wave over 15×15 μm)')
plt.grid(True, alpha=0.3)

# Add some annotations for key features
max_idx = np.argmax(scattering_efficiency)
max_freq = df['Frequency_THz'].iloc[max_idx]
max_eff = scattering_efficiency.iloc[max_idx]

plt.annotate(f'Peak: {max_freq:.1f} THz\nEfficiency: {max_eff:.3f}', 
             xy=(max_freq, max_eff), 
             xytext=(max_freq + 30, max_eff * 0.8),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=12, color='red')

plt.tight_layout()
plt.savefig('hemisphere_scattering_efficiency.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nResults:")
print(f"Peak scattering efficiency: {max_eff:.6f} at {max_freq:.1f} THz")
print(f"Efficiency range: {scattering_efficiency.min():.2e} - {scattering_efficiency.max():.3f}")
print(f"Peak enhancement factor: {max_eff:.1f}×")