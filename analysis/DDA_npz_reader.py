import numpy as np
import matplotlib.pyplot as plt
import re

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
data = np.load(npz_file, allow_pickle=True)
lst = data.files
print("data folders", lst)
folder_paths = data["folder_paths"].tolist()
# for fp in folder_paths:
#     print(fp)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# colors = plt.cm.viridis(np.linspace(0, 1, 9))  # Use a colormap for distinct



full_periodic = np.load("full_periodic_absorption_Cshape1365nmspacing.npy")

p_dis_levels = [0, 100e-9, 200e-9, 300e-9, 400e-9]
seen_p = []
for i, folder_path in enumerate(folder_paths):
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot full periodic on left subplot
    ax1.plot(full_periodic[0], full_periodic[1], color='k', label="Full periodic", linestyle='solid')

    i_5digits = f"{i:05d}"

    # Extract parameters
    l_val, p_val, o_val, m_val = extract_parameters(folder_path)
    
    if o_val == 0:
        print(f"Folder {i}: {folder_path}")

        dipole_count = data[f"sim_{i_5digits}_x_positions"].shape[0]
        print(f"  Dipole count: {dipole_count}")
        area = dipole_count * (1365e-9)**2
        print(area, f"  Area: {area*1e12:.2f} um^2")

        # Visualize dipole positions on right subplot
        ax2.scatter(data[f"sim_{i_5digits}_x_positions"], data[f"sim_{i_5digits}_y_positions"])
        for j in range(1, 7):
            ax2.scatter(data[f"sim_{i_5digits}_x_positions"] + 1365e-9 * j, data[f"sim_{i_5digits}_y_positions"], marker='x', color='gray')
            ax2.scatter(data[f"sim_{i_5digits}_x_positions"] - 1365e-9 * j, data[f"sim_{i_5digits}_y_positions"], marker='x', color='gray')

        freq_list = data[f"sim_{i_5digits}_frequencies"]
        r = data[f"sim_{i_5digits}_r_complex_x"].astype(np.complex128) / area
        t = 1 + r

        R = np.abs(r)**2
        T = np.abs(t)**2
        A = 1 - R - T

        label = None
        if p_val not in seen_p:
            seen_p.append(p_val)
            label = f"p={p_dis_levels[p_val]*1e9:.0f}nm"

        # Plot absorption on left subplot
        ax1.plot(freq_list, A, color=colors[p_val % len(colors)], label=label, marker='o', markersize=4, linestyle='-')

        # Configure subplots
        ax1.legend()
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Absorption')
        ax1.set_title('Absorption Spectra')

        ax2.set_xlim(-20e-6, 20e-6)
        ax2.set_ylim(-20e-6, 20e-6)
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title('Dipole Positions')
        ax2.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(f"absorption_and_dipoles_{i_5digits}_p{p_val}.png", dpi=500)
        plt.show()