import numpy as np
import matplotlib.pyplot as plt
import re

# Set serif font for all matplotlib text
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Computer Modern Roman', 'DejaVu Serif', 'serif']
plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern for math text

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


# npz_file = "/Users/dharper/DDA_simulation_data.npz"
npz_file = "/Users/dharper/DDA_simulation_data_1DperidoicPC.npz"
data = np.load(npz_file, allow_pickle=True)
lst = data.files
# print("data folders", lst)
folder_paths = data["folder_paths"].tolist()
# for fp in folder_paths:
#     print(fp)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# colors = plt.cm.viridis(np.linspace(0, 1, 9))  # Use a colormap for distinct



full_periodic = np.load("full_periodic_absorption_Cshape1365nmspacing.npy")

# Font size settings
axis_label_fontsize = 16    # Size for axis labels
tick_label_fontsize = 12    # Size for tick labels
title_fontsize = 16         # Size for plot title
legend_fontsize = 12        # Size for legend

# Create single figure for all plots
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot full periodic reference
# ax.plot(full_periodic[0], full_periodic[1], color='k', label="Full periodic", linestyle='solid')

p_dis_levels = [0, 100e-9, 200e-9, 300e-9, 400e-9]
seen_p = []
all_frequencies = []  # Collect all frequency data for x-axis limits
for i, folder_path in enumerate(folder_paths):
    i_5digits = f"{i:05d}"

    # Extract parameters
    l_val, p_val, o_val, m_val = extract_parameters(folder_path)
    print(l_val, p_val, o_val, m_val)
    if o_val == 0:
        print(f"Folder {i}: {folder_path}")

        dipole_count = data[f"sim_{i_5digits}_x_positions"].shape[0]
        print(f"  Dipole count: {dipole_count}")
        area = dipole_count * (1365e-9)**2
        print(area, f"  Area: {area*1e12:.2f} um^2")

        freq_list = data[f"sim_{i_5digits}_frequencies"]
        r = data[f"sim_{i_5digits}_r_complex_x"].astype(np.complex128) / area
        t = 1 + r

        R = np.abs(r)**2
        T = np.abs(t)**2
        A = 1 - R - T
        
        # Collect frequency data for x-axis limits
        all_frequencies.extend(freq_list)
        
        # Apply vertical offset based on p_val
        offset = p_val * 0.02
        A_offset = A + offset

        label = None
        if p_val not in seen_p:
            seen_p.append(p_val)
            label = f"p={p_dis_levels[p_val]*1e9:.0f}nm"

        # Plot absorption on single plot with offset
        ax.plot(freq_list*1e-12, A_offset, color=colors[p_val % len(colors)], label=label, marker='o', markersize=5, linestyle='-', linewidth=4)

# # Plot full periodic reference
# ax.plot(full_periodic[0], full_periodic[1], color='k', label="Full periodic", linestyle='solid')

# Configure plot
ax.legend(fontsize=legend_fontsize, loc='upper right')
ax.set_xlabel('Frequency (THz)', fontsize=axis_label_fontsize)
ax.set_ylabel('Absorption (displaced units)', fontsize=axis_label_fontsize)
ax.set_title('PC Absorption spectra vs. Disorder', fontsize=title_fontsize)

# Set tick label font sizes
ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)

# Set y-axis to start at zero
ax.set_ylim(bottom=0)

# Set x-axis limits based on data range
if all_frequencies:
    freq_min = min(all_frequencies) * 1e-12
    freq_max = max(all_frequencies) * 1e-12
    ax.set_xlim(freq_min, freq_max)

# Add grid for better readability
ax.grid(True, alpha=0.0)

plt.tight_layout()
plt.savefig("absorption_spectra_stacked.png", dpi=500)
plt.show()