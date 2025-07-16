import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from CSV files - note COMSOL data is tab-separated
comsol_data = pd.read_csv('/Users/dharper/Documents/DDA_C/figures/COMSOL_perturbations/COMSOL_doubly_perturbed_100nm.csv', sep='\t')
dda_data_complex = pd.read_csv('/Users/dharper/Documents/DDA_C/figures/COMSOL_perturbations/doubly_perturbed_2x2_100nm/cdm_input_0/reflection_transmission_complex.csv')
# dda_data = pd.read_csv('/Users/dharper/Documents/DDA_C/figures/COMSOL_perturbations/doubly_perturbed_2x2_0nm/cdm_input_0/reflection_transmission.csv')

# Convert perturbed data frequency from Hz to the same scale as COMSOL data
dda_data_complex['frequency'] = dda_data_complex['frequency'] / 1e12
r = np.complex128(dda_data_complex['r'].to_numpy())
t = np.complex128(dda_data_complex['t'].to_numpy())

# Create figure with two subplots
ax = plt.gca()

# Plot Reflection data
ax.plot(comsol_data['frequency'], comsol_data['R'], label='COMSOL Unperturbed', linewidth=2)
ax.plot(dda_data_complex['frequency'], np.abs(r)**2, label='DDA Unperturbed', linewidth=2, linestyle='--')

ax.set_xlabel('Frequency (THz)')
ax.set_ylabel('Reflection and Transmission')
ax.set_title('Reflection and Transmission Comparison')
ax.legend()
ax.grid(True)

# Plot Transmission data
ax.plot(comsol_data['frequency'], comsol_data['T'], label='COMSOL Unperturbed', linewidth=2)
ax.plot(dda_data_complex['frequency'], np.abs(t)**2, label='DDA Unperturbed', linewidth=2, linestyle='--')

plt.tight_layout()
plt.show()