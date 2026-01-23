import numpy as np
import matplotlib.pyplot as plt
import re

# Set serif font for all matplotlib text
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Computer Modern Roman', 'DejaVu Serif', 'serif']
plt.rcParams['mathtext.fontset'] = 'cm'

def extract_parameters(folder_path):
    pattern = r'l(\d+)_p(\d+)_o(\d+)_m(\d+)'
    match = re.search(pattern, folder_path)
    if match:
        return tuple(int(match.group(i)) for i in range(1, 5))
    return None, None, None, None


npz_file = "/Users/dharper/DDA_simulation_data_1DperidoicPC.npz"
data = np.load(npz_file, allow_pickle=True)

folder_paths = data["folder_paths"].tolist()

# Font sizes
axis_label_fontsize = 12
tick_label_fontsize = 8
title_fontsize = 12
legend_fontsize = 8

p_dis_levels = [0, 100e-9, 200e-9, 300e-9, 400e-9]

# Create stacked subplots with tight spacing
fig, axes = plt.subplots(
    len(p_dis_levels),
    1,
    figsize=(9, 5),
    sharex=True,
    gridspec_kw=dict(hspace=0.05)
)

seen_p = []
all_frequencies = []

for i, folder_path in enumerate(folder_paths):
    i_5digits = f"{i:05d}"

    l_val, p_val, o_val, m_val = extract_parameters(folder_path)
    if o_val != 0:
        continue

    dipole_count = data[f"sim_{i_5digits}_x_positions"].shape[0]
    area = dipole_count * (1365e-9)**2

    freq_list = data[f"sim_{i_5digits}_frequencies"]
    r = data[f"sim_{i_5digits}_r_complex_x"].astype(np.complex128) / area
    t = 1 + r

    R = np.abs(r)**2
    T = np.abs(t)**2
    A = 1 - R - T

    all_frequencies.extend(freq_list)

    ax = axes[p_val]

    label = None
    if p_val not in seen_p:
        seen_p.append(p_val)
        label = f"p={p_dis_levels[p_val]*1e9:.0f}nm"

    ax.plot(
        freq_list * 1e-12,
        A,
        color=(90/255, 100/255, 210/255),
        marker='o',
        markersize=3,
        linewidth=2,
        label=label
    )

    ax.legend(fontsize=legend_fontsize, loc='upper right')
    ax.set_ylim(0, 0.03)
    ax.grid(False)

    # Show full y-ticks on every subplot
    ax.tick_params(axis='y', labelsize=tick_label_fontsize)

# X-ticks only on bottom
for ax in axes[:-1]:
    ax.tick_params(axis='x', bottom=False, labelbottom=False)

axes[-1].tick_params(axis='x', labelsize=tick_label_fontsize)

# Shared labels
fig.supxlabel('Frequency (THz)', fontsize=axis_label_fontsize)
fig.supylabel('Absorption', fontsize=axis_label_fontsize)

fig.suptitle('PC Absorption spectra vs. Disorder', fontsize=title_fontsize)

# Shared x-limits
if all_frequencies:
    axes[-1].set_xlim(
        min(all_frequencies) * 1e-12,
        max(all_frequencies) * 1e-12
    )

plt.savefig("absorption_spectra_stacked.pdf", dpi=500, bbox_inches="tight")
plt.show()
