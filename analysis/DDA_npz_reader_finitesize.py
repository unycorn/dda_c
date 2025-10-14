import numpy as np
import matplotlib.pyplot as plt
import re

Z0 = 376.730313668  # Impedance of free space in ohms
eps0 = 8.854187817e-12  # Permittivity of free space in F/m

def alpha_func_U(freq):
    f0 = 3.055409e+14
    hw = 2.002573e+13
    ee_A = 1.057429e+08
    ee_B = 1.816242e-23
    ee_C = 1.856486e-36

    alpha_scaled = ee_A / (f0**2 - freq**2 - 1j * freq * hw) + ee_B + ee_C * freq
    return alpha_scaled * eps0

def extract_parameters(folder_path):
    """
    Extract l, p, o, m parameters from folder path string
    Expected format: ...l0_p4_o0_m0...
    """
    # Use regex to find the pattern l<number>_p<number>_o<number>_m<number>
    pattern = r'l(\d+)_p(\d+)_o(\d+)_m(\d+)'
    match = re.search(pattern, folder_path)
    
    if match:
        l_val = int(match.group(1))
        p_val = int(match.group(2))
        o_val = int(match.group(3))
        m_val = int(match.group(4))
        return l_val, p_val, o_val, m_val
    else:
        return None, None, None, None


npz_file = "/Users/dharper/DDA_simulation_data.npz"
# npz_file = "/Users/dharper/DDA_simulation_data_1DperidoicPC.npz"
data = np.load(npz_file, allow_pickle=True)
lst = data.files
print("data folders", lst)
folder_paths = data["folder_paths"].tolist()
# for fp in folder_paths:
#     print(fp)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# colors = plt.cm.viridis(np.linspace(0, 1, 9))  # Use a colormap for distinct


p_dis_levels = [0, 100e-9, 200e-9, 300e-9, 400e-9]
seen_p = []
for i, folder_path in enumerate(folder_paths):
    # Create single figure
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    i_5digits = f"{i:05d}"

    # Extract parameters
    l_val, p_val, o_val, m_val = extract_parameters(folder_path)
    
    if o_val == 0:
        print(f"Folder {i}: {folder_path}")

        dipole_count = data[f"sim_{i_5digits}_x_positions"].shape[0]
        print(f"  Dipole count: {dipole_count}")
        spacing = 100e-9
        area = dipole_count * (spacing)**2
        print(area, f"  Area: {area*1e12:.2f} um^2")



        freq_list = data[f"sim_{i_5digits}_frequencies"]
        r = data[f"sim_{i_5digits}_r_complex_x"].astype(np.complex128) / area
        t = 1 + r

        R = np.abs(r)**2
        T = np.abs(t)**2
        A = 1 - R - T

        ppP = 1/(2*Z0)
        A2 = -np.pi * freq_list * np.imag(1/alpha_func_U(freq_list)) * (np.abs(data[f"sim_{i_5digits}_polarizations_ex"])**2).sum(axis=1) / ppP

        Cext = np.pi * freq_list * np.imag(data[f"sim_{i_5digits}_polarizations_ex"]).sum(axis=1) / ppP


        label = None
        if p_val not in seen_p:
            seen_p.append(p_val)
            label = f"p={p_dis_levels[p_val]*1e9:.0f}nm"

        # Plot absorption on left subplot
        ax1.plot(freq_list, A, color=colors[p_val % len(colors)], label=label, marker='o', markersize=4, linestyle='-')

        ax1.plot(freq_list, np.real(r), color=colors[p_val % len(colors)], label=label, marker='.', markersize=4, linestyle='-')
        # ax1.plot(freq_list, np.imag(r), color=colors[p_val % len(colors)], label=label, marker='.', markersize=4, linestyle='--')

        # ax1.plot(freq_list, Cext, color=colors[p_val % len(colors)], linestyle='--', alpha=0.5)

        # Configure plot
        ax1.legend()
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Absorption')
        ax1.set_title('Absorption Spectra')

        plt.tight_layout()
        plt.savefig(f"absorption_and_dipoles_{i_5digits}_p{p_val}.png", dpi=500)
        plt.show()