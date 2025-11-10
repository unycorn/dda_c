import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

# Create 3D figure for waterfall plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot full periodic reference
# ax.plot(full_periodic[0], full_periodic[1], color='k', label="Full periodic", linestyle='solid')

p_dis_levels = [0, 100e-9, 200e-9, 300e-9, 400e-9]
seen_p = []
plot_data = []    # Store data for plotting in reverse order

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

        label = None
        if p_val not in seen_p:
            seen_p.append(p_val)
            label = f"p={p_dis_levels[p_val]*1e9:.0f}nm"
            
            # Store data for plotting (will plot in reverse order for proper z-ordering)
            y_pos = p_val * 0.5  # Space curves apart for waterfall effect
            plot_data.append({
                'freq': freq_list,
                'absorption': A,
                'y_pos': y_pos,
                'color': colors[p_val % len(colors)],
                'label': label
            })

# Plot in reverse order to fix z-order rendering (back to front)
for data_point in reversed(plot_data):
    # Create constant Y values for the waterfall effect
    y_vals = [data_point['y_pos']] * len(data_point['freq'])
    
    # Convert frequency from Hz to THz
    freq_thz = data_point['freq'] / 1e12
    
    # Plot the 3D line
    ax.plot(freq_thz, y_vals, data_point['absorption'], 
            color=data_point['color'], linewidth=3, label=data_point['label'])

# Plot full periodic reference at the front (y_pos = -0.5 to separate it)
full_periodic_y = [0] * len(full_periodic[0])
# Convert full periodic frequency from Hz to THz
full_periodic_freq_thz = full_periodic[0] / 1e12
ax.plot(full_periodic_freq_thz, full_periodic_y, full_periodic[1], 
        color='k', linewidth=2, label="Full periodic", linestyle='dotted')

# Configure 3D plot
ax.set_xlabel('Frequency (THz)', labelpad=15)
ax.set_ylabel('Disorder Level $\sigma_{xy}$ [nm]', labelpad=15)
ax.set_zlabel('Absorption', labelpad=10)
ax.set_title('PC Absorption Spectra - Waterfall Plot')
ax.legend()

# Set appropriate limits
if plot_data:
    # Get frequency range from first dataset (convert to THz)
    freq_min_thz, freq_max_thz = np.min(plot_data[0]['freq']) / 1e12, np.max(plot_data[0]['freq']) / 1e12
    ax.set_xlim3d(freq_min_thz, freq_max_thz)
    
    # Set Y limits based on disorder levels
    ax.set_ylim3d(-0.5, max([d['y_pos'] for d in plot_data]) + 0.5)
    
    # Set Z limits for absorption
    ax.set_zlim3d(0, 0.025)
    
    # Set custom Y-tick labels to show actual disorder levels in nanometers
    y_tick_positions = [0] + [i * 0.5 for i in range(1, len(p_dis_levels))]  # Include position 0 for full periodic
    y_tick_labels = ['0 nm'] + [f'{level*1e9:.0f} nm' for level in p_dis_levels[1:]]  # Start with 0 nm instead of "Full Periodic"
    
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)

plt.tight_layout()
plt.savefig("absorption_spectra_waterfall.png", dpi=500)
plt.show()