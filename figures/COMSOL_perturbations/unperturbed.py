import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV files - note COMSOL data is tab-separated
comsol_data = pd.read_csv('/Users/dharper/Documents/DDA_C/figures/COMSOL_perturbations/COMSOL_unperturbed.csv', sep='\t')
perturbed_data = pd.read_csv('/Users/dharper/Documents/DDA_C/figures/COMSOL_perturbations/doubly_perturbed_0nm/cdm_input_0/reflection_transmission.csv')

# Convert perturbed data frequency from Hz to the same scale as COMSOL data
perturbed_data['frequency'] = perturbed_data['frequency'] / 1e12

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot Reflection data
ax1.plot(comsol_data['frequency'], comsol_data['R'], label='COMSOL Unperturbed', linewidth=2)
ax1.plot(perturbed_data['frequency'], perturbed_data['R'], label='DDA Unperturbed', linewidth=2, linestyle='--')
ax1.set_xlabel('Frequency (THz)')
ax1.set_ylabel('Reflection')
ax1.set_title('Reflection Comparison')
ax1.legend()
ax1.grid(True)

# Plot Transmission data
ax2.plot(comsol_data['frequency'], comsol_data['T'], label='COMSOL Unperturbed', linewidth=2)
ax2.plot(perturbed_data['frequency'], perturbed_data['T'], label='DDA Unperturbed', linewidth=2, linestyle='--')
ax2.set_xlabel('Frequency (THz)')
ax2.set_ylabel('Transmission')
ax2.set_title('Transmission Comparison')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()