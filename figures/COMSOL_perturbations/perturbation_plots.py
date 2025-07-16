import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
Z0 = 376.730313668  # Ohms, characteristic impedance of free space
I = 1j

def compare_comsol_dda(comsol_data, dda_data_complex, name):
    # Convert perturbed data frequency from Hz to the same scale as COMSOL data
    dda_data_complex['frequency'] = dda_data_complex['frequency'] / 1e12
    dda_r = np.complex128(dda_data_complex['r'].to_numpy())
    dda_t = np.complex128(dda_data_complex['t'].to_numpy())

    ax = plt.gca()

    # Plot Reflection data
    ax.plot(dda_data_complex['frequency'], np.abs(dda_r)**2, label='DDA Unperturbed', linewidth=2, color='red')
    ax.plot(comsol_data['frequency'], comsol_data['R'], label='COMSOL Unperturbed', linewidth=2, linestyle='--', color='blue')

    # Plot Transmission data
    ax.plot(dda_data_complex['frequency'], np.abs(dda_t)**2, linewidth=2, color='red')
    ax.plot(comsol_data['frequency'], comsol_data['T'], linewidth=2, linestyle='--', color='blue')

    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('Reflection and Transmission')
    ax.set_title(f'Reflection and Transmission Comparison {name}')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot sheet retrieval data
    ax = plt.gca()

    ax.plot(dda_data_complex['frequency'], 1 - np.abs(dda_r)**2 - np.abs(dda_t)**2, label='DDA Unperturbed', linewidth=2)
    ax.plot(comsol_data['frequency'], 1 - comsol_data['R'] - comsol_data['T'], label='COMSOL Unperturbed', linewidth=2, linestyle='--')

    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('1 - R - T')
    ax.set_title(f'Reflection and Transmission Comparison {name}')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot sheet retrieval data
    ax = plt.gca()

    sigma_se = (2/Z0) * (1 - dda_r - dda_t) / (1 + dda_r + dda_t)
    sigma_sm = (2*Z0) * (1 + dda_r - dda_t) / (1 - dda_r + dda_t)

    ax.plot(dda_data_complex['frequency'], np.real(sigma_se), label='DDA $\\sigma_{se}$ real', linewidth=2, color='blue', linestyle='-')
    ax.plot(dda_data_complex['frequency'], np.imag(sigma_se), label='DDA $\\sigma_{se}$ imag', linewidth=2, color='blue', linestyle='--')

    ax.plot(dda_data_complex['frequency'], np.real(sigma_sm), label='DDA $\\sigma_{sm}$ real', linewidth=2, color='red', linestyle='-')
    ax.plot(dda_data_complex['frequency'], np.imag(sigma_sm), label='DDA $\\sigma_{sm}$ imag', linewidth=2, color='red', linestyle='--')
    
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('$\\sigma_{se}$, $\\sigma_{sm}$')
    ax.set_title(f'Sheet Retrieval Comparison {name}')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    ax = plt.gca()

    chi_se = sigma_se / (-I * 2 * np.pi * dda_data_complex['frequency'])
    chi_sm = sigma_sm / (-I * 2 * np.pi * dda_data_complex['frequency'])

    ax.plot(dda_data_complex['frequency'], np.real(chi_se), label='DDA $\\chi{se}$ real', linewidth=2, color='blue', linestyle='-')
    ax.plot(dda_data_complex['frequency'], np.imag(chi_se), label='DDA $\\chi{se}$ imag', linewidth=2, color='blue', linestyle='--')

    ax.plot(dda_data_complex['frequency'], np.real(chi_sm), label='DDA $\\chi{sm}$ real', linewidth=2, color='red', linestyle='-')
    ax.plot(dda_data_complex['frequency'], np.imag(chi_sm), label='DDA $\\chi{sm}$ imag', linewidth=2, color='red', linestyle='--')
    
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('$\\chi{se}$, $\\chi{sm}$')
    ax.set_title(f'Sheet Retrieval Comparison {name}')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# Load the data from CSV files - note COMSOL data is tab-separated
comsol_data_unperturbed = pd.read_csv('/Users/dharper/Documents/DDA_C/figures/COMSOL_perturbations/COMSOL_unperturbed.csv', sep='\t')
dda_data_complex_unperturbed = pd.read_csv('/Users/dharper/Documents/DDA_C/figures/COMSOL_perturbations/doubly_perturbed_2x2_0nm/cdm_input_0/reflection_transmission_complex.csv')

comsol_data_singly = pd.read_csv('/Users/dharper/Documents/DDA_C/figures/COMSOL_perturbations/COMSOL_singly_perturbed_100nm.csv', sep='\t')
dda_data_singly = pd.read_csv('/Users/dharper/Documents/DDA_C/figures/COMSOL_perturbations/singly_perturbed_2x2_100nm/cdm_input_0/reflection_transmission_complex.csv')

comsol_data_doubly = pd.read_csv('/Users/dharper/Documents/DDA_C/figures/COMSOL_perturbations/COMSOL_doubly_perturbed_100nm.csv', sep='\t')
dda_data_complex_doubly = pd.read_csv('/Users/dharper/Documents/DDA_C/figures/COMSOL_perturbations/doubly_perturbed_2x2_100nm/cdm_input_0/reflection_transmission_complex.csv')


compare_comsol_dda(comsol_data_unperturbed, dda_data_complex_unperturbed, "Unperturbed")
compare_comsol_dda(comsol_data_singly, dda_data_singly, "Singly Perturbed")
compare_comsol_dda(comsol_data_doubly, dda_data_complex_doubly, "Doubly Perturbed")