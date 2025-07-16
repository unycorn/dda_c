import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
Z0 = 376.730313668  # Ohms, characteristic impedance of free space

script_path = os.path.dirname(os.path.abspath(__file__))
DDA_folder = os.path.join(script_path, 'DDA_double_perturbation_sweep')
DDA_folder_list = os.listdir(DDA_folder)
DDA_folder_list.sort()  # Sort the folder list for consistent plotting

def convert_complex(x):
    # Convert string with 'i' to complex number
    if isinstance(x, str):
        return np.complex128(x.replace('i', 'j'))
    return x

file_paths = [
    os.path.join(script_path, 'COMSOL0.csv'),
    os.path.join(script_path, 'COMSOL40-80.csv'),
    os.path.join(script_path, 'COMSOL100.csv')
]
dfs = []
for file_path in file_paths:
    try:
        # Read CSV as strings first
        df = pd.read_csv(file_path, sep='\t', dtype=str)
        # Convert numeric columns to float
        df['frequency'] = df['frequency'].astype(float)
        df['perturbation'] = df['perturbation'].astype(float)  # Add this line
        # Convert complex columns
        for col in ['reflection', 'transmission']:
            df[col] = df[col].apply(convert_complex)
        dfs.append(df)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        continue

# Combine all dataframes
COMSOL_data = pd.concat(dfs, ignore_index=True)
# Sort COMSOL data by frequency for consistent plotting
COMSOL_data.sort_values(by='frequency', inplace=True)

DDA_folder_list = os.listdir(DDA_folder)
DDA_folder_list.sort()  # Sort the folder list for consistent plotting

for i, folder in enumerate(DDA_folder_list):
    print(f"Processing folder: {folder}")
    spacing_nm = float(folder.split('_')[-1][:-2])
    csv_path = os.path.join(DDA_folder, folder, 'cdm_input_0/reflection_transmission_complex.csv')

    # Read tab-separated data with headers
    data = pd.read_csv(csv_path, sep=',')
    
    # Convert r and t columns to complex numbers
    data['r'] = data['r'].apply(lambda x: np.complex128(x))
    data['t'] = data['t'].apply(lambda x: np.complex128(x))
    
    # plt.plot(data['frequency'], np.abs(data['r'])**2, label=f'R, Spacing {spacing_nm} nm', color=f"C{i}")
    # plt.plot(data['frequency'], np.abs(data['t'])**2, label=f'T, Spacing {spacing_nm} nm', color=f"C{i}")

    COMSOL_data_0 = COMSOL_data[np.isclose(COMSOL_data["perturbation"], spacing_nm*1e-9, atol=1e-10)]  # Convert spacing_nm to meters for comparison
    print(len(COMSOL_data_0), "COMSOL data length for spacing", spacing_nm)

    # plt.plot(data['frequency'], np.angle(data['r']), label=f'R, Spacing {spacing_nm} nm', color=f"red")
    # plt.plot(data['frequency'], np.angle(data['t']), label=f'T, Spacing {spacing_nm} nm', color=f"blue")

    # plt.plot(COMSOL_data_0['frequency']*1e12, np.angle(COMSOL_data_0['reflection']), label=f'R, Spacing {spacing_nm} nm', color=f"red")
    # plt.plot(COMSOL_data_0['frequency']*1e12, np.angle(COMSOL_data_0['transmission']), label=f'T, Spacing {spacing_nm} nm', color=f"blue")

    plt.plot(data['frequency'], np.real(data['r']), label=f'Re[r] DDA, Spacing {spacing_nm} nm', color=f"red", linestyle='-')
    plt.plot(data['frequency'], np.imag(data['r']), label=f'Im[r] DDA, Spacing {spacing_nm} nm', color=f"red", linestyle='--')
    plt.plot(COMSOL_data_0['frequency']*1e12, np.real(COMSOL_data_0['reflection']), label=f'Re[r] COMSOL, Spacing {spacing_nm} nm', color=f"blue", linestyle='-')
    plt.plot(COMSOL_data_0['frequency']*1e12, np.imag(COMSOL_data_0['reflection']), label=f'Im[r] COMSOL, Spacing {spacing_nm} nm', color=f"blue", linestyle='--')
    plt.legend()
    plt.savefig(f"reflection_comparison_spacing_{spacing_nm}nm.png")
    plt.show()

    plt.plot(data['frequency'], np.real(data['t']), label=f'Re[t] DDA, Spacing {spacing_nm} nm', color=f"red", linestyle='-')
    plt.plot(data['frequency'], np.imag(data['t']), label=f'Im[t] DDA, Spacing {spacing_nm} nm', color=f"red", linestyle='--')
    plt.plot(COMSOL_data_0['frequency']*1e12, np.real(COMSOL_data_0['transmission']), label=f'Re[t] COMSOL, Spacing {spacing_nm} nm', color=f"blue", linestyle='-')
    plt.plot(COMSOL_data_0['frequency']*1e12, np.imag(COMSOL_data_0['transmission']), label=f'Im[t] COMSOL, Spacing {spacing_nm} nm', color=f"blue", linestyle='--')
    plt.legend()
    plt.savefig(f"transmission_comparison_spacing_{spacing_nm}nm.png")
    plt.show()

    DDA_sigma_se = (2/Z0) * (1 - data['r'] - data['t']) / (1 + data['r'] + data['t'])
    DDA_sigma_sm = (2*Z0) * (1 + data['r'] - data['t']) / (1 - data['r'] + data['t'])
    
    COMSOL_sigma_se = (2/Z0) * (1 - COMSOL_data_0['reflection'] - COMSOL_data_0['transmission']) / (1 + COMSOL_data_0['reflection'] + COMSOL_data_0['transmission'])
    COMSOL_sigma_sm = (2*Z0) * (1 + COMSOL_data_0['reflection'] - COMSOL_data_0['transmission']) / (1 - COMSOL_data_0['reflection'] + COMSOL_data_0['transmission'])

    print(data['r'], COMSOL_data_0['reflection'], COMSOL_sigma_se, DDA_sigma_se)

    plt.plot(data['frequency'], np.real(DDA_sigma_se), label=f'Re[$\\sigma_{{se}}$] DDA, Spacing {spacing_nm} nm', color=f"red", linestyle='-')
    plt.plot(data['frequency'], np.imag(DDA_sigma_se), label=f'Im[$\\sigma_{{se}}$] DDA, Spacing {spacing_nm} nm', color=f"red", linestyle='--')
    plt.plot(COMSOL_data_0['frequency']*1e12, np.real(COMSOL_sigma_se), label=f'Re[$\\sigma_{{se}}$] COMSOL, Spacing {spacing_nm} nm', color=f"blue", linestyle='-')
    plt.plot(COMSOL_data_0['frequency']*1e12, np.imag(COMSOL_sigma_se), label=f'Re[$\\sigma_{{se}}$] COMSOL, Spacing {spacing_nm} nm', color=f"blue", linestyle='--')
    plt.legend()
    plt.savefig(f"sigma_se_comparison_spacing_{spacing_nm}nm.png")
    plt.show()


    # plt.plot(COMSOL_data_0['frequency']*1e12, np.angle(COMSOL_data_0['reflection']), label=f'R, Spacing {spacing_nm} nm', color=f"red")
    # plt.plot(COMSOL_data_0['frequency']*1e12, np.angle(COMSOL_data_0['transmission']), label=f'T, Spacing {spacing_nm} nm', color=f"blue")
# plt.legend()
# plt.show()

