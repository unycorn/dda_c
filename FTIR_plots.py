import os
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import re

# Find all unique m values in the directory structure
base_dir = "./csv_inputs"  # Changed to local directory
if not os.path.exists(base_dir):
    print(f"Error: {base_dir} does not exist")
    exit(1)

# Get all subdirectories and extract m values
m_values = set()
pattern = re.compile(r'p\d+_o\d+_m(\d+)')
for dirname in os.listdir(base_dir):
    match = pattern.match(dirname)
    if match:
        m_values.add(int(match.group(1)))

if not m_values:
    print("No valid directories found")
    exit(1)

print(f"Found m values: {sorted(m_values)}")

# Function to process data for a specific m value
def process_m_value(m_val):
    print(f"\nProcessing m value: {m_val}")
    all_refls = []
    all_trans = []
    
    plist = [0, 25e-9, 50e-9, 75e-9, 100e-9]  # meters
    olist = [0, 10, 20, 50, 10000]  # radians
    
    # First pass to collect all reflection and transmission values
    for p in range(5):
        for o in range(5):
            folder_name = f"p{p}_o{o}_m{m_val}"
            folder_path = os.path.join(base_dir, folder_name)
            
            if not os.path.exists(folder_path):
                print(f"Skipping non-existent folder: {folder_name}")
                continue
            
            print(f"Processing folder: {folder_name}")
            data_found = False
                
            for i in range(0, 20):
                subfolder = os.path.join(folder_path, f"cdm_input_{i}")
                if not os.path.exists(subfolder):
                    continue
                
                data = []
                for fname in ["freq_r_t.txt", "freq_r_t_augment.txt"]:
                    file_path = os.path.join(subfolder, fname)
                    if os.path.isfile(file_path):
                        try:
                            with open(file_path, "r") as f:
                                lines = f.readlines()
                            data.extend([eval(line.strip().rstrip(',')) for line in lines])
                            data_found = True
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")
                
                if data:
                    _, refls, trans = zip(*data)
                    all_refls.extend(refls)
                    all_trans.extend(trans)
            
            if not data_found:
                print(f"No data found in folder: {folder_name}")
    
    if not all_refls or not all_trans:
        print(f"No data found for m={m_val}")
        return
    
    # Calculate ranges
    refl_max = max(all_refls)
    trans_min = min(all_trans)
    
    print(f"For m={m_val}: Reflection max = {refl_max:.3f}, Transmission min = {trans_min:.3f}")
    
    # Create three different plots with different y-axis limits
    ylims = [
        (-0.1, 1.1, 'RT'),  # Full range for both R and T
        (0, refl_max * 1.1, 'R'),  # Range for reflectance only
        (trans_min * 0.9, 1.0, 'T')  # Range for transmittance only
    ]
    
    for ylim, plot_type in ylims:
        # Reset the figure for each plot type
        fig, axes = plt.subplots(5, 5, figsize=(20, 20), sharex=True, sharey=True)
        
        for p in range(5):
            for o in range(5):
                folder_name = f"p{p}_o{o}_m{m_val}"
                folder_path = os.path.join(base_dir, folder_name)
                ax = axes[4 - p, o]
                ax.set_ylim(ylim)
                
                if not os.path.exists(folder_path):
                    continue
                
                for i in range(0, 20):
                    subfolder = os.path.join(folder_path, f"cdm_input_{i}")
                    if not os.path.exists(subfolder):
                        continue
                        
                    data = []
                    for fname in ["freq_r_t.txt", "freq_r_t_augment.txt"]:
                        file_path = os.path.join(subfolder, fname)
                        if os.path.isfile(file_path):
                            try:
                                with open(file_path, "r") as f:
                                    lines = f.readlines()
                                data.extend([eval(line.strip().rstrip(',')) for line in lines])
                            except Exception as e:
                                print(f"Error reading {file_path}: {e}")
                    
                    if not data:
                        continue
                    
                    freqs, refls, trans = zip(*data)
                    sort_idx = np.argsort(freqs)
                    freqs = np.array(freqs)[sort_idx]
                    refls = np.array(refls)[sort_idx]
                    trans = np.array(trans)[sort_idx]
                    
                    if plot_type == 'RT':
                        ax.plot(freqs, refls, 'b-', alpha=0.3)
                        ax.plot(freqs, trans, 'r-', alpha=0.3)
                    elif plot_type == 'R':
                        ax.plot(freqs, refls, 'b-', alpha=0.3)
                    else:  # plot_type == 'T'
                        ax.plot(freqs, trans, 'r-', alpha=0.3)
                
                if o == 0:
                    ax.set_ylabel(f'p={plist[p]}m')
                if p == 4:
                    ax.set_xlabel(f'o={olist[o]}°')
        
        plt.suptitle(f'M={m_val} {plot_type} Plot')
        plt.tight_layout()
        output_file = os.path.join(base_dir, f'm{m_val}_{plot_type.lower()}_plot.png')
        plt.savefig(output_file)
        print(f"Saved plot to {output_file}")
        plt.close()

# Process each m value
for m_val in sorted(m_values):
    process_m_value(m_val)