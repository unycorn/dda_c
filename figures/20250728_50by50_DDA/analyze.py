import numpy as np
import os
import matplotlib.pyplot as plt
import re

# Load the numpy data file
data_file = "/Users/dharper/Documents/DDA_C/figures/20250728_50by50_DDA/20250728_DDA_rtxy_values.npz"
data = np.load(data_file, allow_pickle=True)

# Display available arrays in the npz file
print("Available arrays in the file:")
for key in data.files:
    print(f"  {key}: shape {data[key].shape}, dtype {data[key].dtype}")

# Access the data
folder_paths = data['folder_paths']
results_x_values = data['results_x']
results_y_values = data['results_y']

# Define color and line style mappings
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
line_styles = ['-', '-', '-', '-', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (3, 5, 1, 5)), (0, (1, 1)), (0, (3, 10, 1, 10)), (0, (5, 1))]

# Create 5x5 subplot grid
fig, axes = plt.subplots(5, 5, figsize=(20, 20))
# fig.suptitle('Reflection and Transmission Coefficients - 5x5 Grid', fontsize=16)

# Function to parse folder name parameters
def parse_folder_name(folder_path):
    # Extract the folder name part that contains the parameters
    folder_name = folder_path.split('/')[-2] if '/' in folder_path else folder_path
    
    # Use regex to find l, p, o, m values
    l_match = re.search(r'l(\d+)', folder_name)
    p_match = re.search(r'p(\d+)', folder_name)
    o_match = re.search(r'o(\d+)', folder_name)
    m_match = re.search(r'm(\d+)', folder_name)
    
    l_val = int(l_match.group(1)) if l_match else 0
    p_val = int(p_match.group(1)) if p_match else 0
    o_val = int(o_match.group(1)) if o_match else 0
    m_val = int(m_match.group(1)) if m_match else 0
    
    return l_val, p_val, o_val, m_val

spatial_disorder_degrees = [0, 25, 50, 75, 100]  # nm
orientational_disorder_degrees = [0, 10, 20, 50, '\\infty']  # deg
l_value_labels = ['square', 'triangular', 'amman-beenker']

current_l_val = 2
print(f"Processing l={current_l_val}")

# Process each folder
for i, folder in enumerate(folder_paths):
    print(f"Folder {i}: {folder}")
    
    # Parse parameters from folder name
    l_val, p_val, o_val, m_val = parse_folder_name(folder)
    print(f"  Parameters: l={l_val}, p={p_val}, o={o_val}, m={m_val}")
    if l_val != current_l_val:
        continue
    if m_val > 2:
        continue
    
    # Determine subplot position (p=y, o=x) - flip y-axis by using (4-p_val)
    if p_val < 5 and o_val < 5:  # Ensure we're within the 5x5 grid
        ax = axes[4-p_val, o_val]
        
        # Process the data
        freq_r_t_X = np.array(results_x_values[i], dtype=np.complex128)
        freq_r_t_Y = np.array(results_y_values[i], dtype=np.complex128)
        freq = np.real(freq_r_t_X[:, 0])
        r_x = freq_r_t_X[:, 1]
        t_x = freq_r_t_X[:, 2]
        r_y = freq_r_t_Y[:, 1]
        t_y = freq_r_t_Y[:, 2]
        
        # Determine color and line style
        color = colors[m_val % len(colors)]
        line_style = line_styles[l_val % len(line_styles)]
        
        # Plot on the appropriate subplot
        # ax.plot(freq, np.abs(r_x)**2, color=color, linestyle=line_style, 
        #         label=f'R_x (l{l_val}m{m_val})', alpha=0.3)
        # ax.plot(freq, np.abs(t_x)**2, color=color, linestyle=line_style, 
        #         label=f'T_x (l{l_val}m{m_val})', alpha=0.3)
        
        ax.plot(freq, 1 - np.abs(r_x)**2 - np.abs(t_x)**2, color=color, linestyle=line_style, 
                label=f'A_x (l{l_val}m{m_val})', alpha=0.3)
        
        # Customize subplot
        ax.set_title(f'$\\sigma_{{x,y}}={spatial_disorder_degrees[p_val]}$ nm, $\\sigma_\\theta={orientational_disorder_degrees[o_val]}$ deg', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([freq.min(), freq.max()])
        
        # Add labels only to edge subplots
        if p_val == 0:  # Bottom row (after flipping, p=0 is now at the bottom)
            ax.set_xlabel('Frequency (Hz)', fontsize=8)
        if o_val == 0:  # Left column
            ax.set_ylabel('Magnitude (1 - r*r - t*t)', fontsize=8)
    else:
        print(f"  Warning: p={p_val} or o={o_val} outside 5x5 grid, skipping")

# Adjust layout and show
plt.tight_layout()
# plt.savefig('reflection_transmission_coefficients_5x5_grid.pdf')
plt.savefig(f'absorption_coefficients_5x5_grid_cshape_{l_value_labels[current_l_val]}.pdf')
plt.show()