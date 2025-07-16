import os
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Generate chi plots from reflection/transmission data')
    parser.add_argument('base_dir', help='Base directory containing the csv input folders')
    return parser.parse_args()

base_dir = "./csv_inputs"

# Get all subdirectories and extract l and m values
l_values = set()
m_values = set()
pattern = re.compile(r'l(\d+)_p\d+_o\d+_m(\d+)')
for dirname in os.listdir(base_dir):
    match = pattern.match(dirname)
    if match:
        l_values.add(int(match.group(1)))
        m_values.add(int(match.group(2)))

def compute_global_ranges():
    """Compute global min/max ranges for R and T based on averaged curves."""
    all_avg_refls = []
    all_avg_trans = []
    
    # Iterate through all possible combinations to collect data
    for l_val in sorted(l_values):
        for m_val in sorted(m_values):
            for p in range(5):
                for o in range(5):
                    folder_name = f"l{l_val}_p{p}_o{o}_m{m_val}"
                    folder_path = os.path.join(base_dir, folder_name)
                    
                    if not os.path.exists(folder_path):
                        continue
                    
                    # Collect all curves for this configuration
                    config_refls = []
                    config_trans = []
                    freqs = None
                    
                    for i in range(0, 10):
                        subfolder = os.path.join(folder_path, f"cdm_input_{i}")
                        if not os.path.exists(subfolder):
                            continue
                        
                        file_path = os.path.join(subfolder, "reflection_transmission_complex.csv")
                        if not os.path.isfile(file_path):
                            continue
                            
                        # Read CSV with complex amplitudes
                        df = pd.read_csv(file_path)
                        freqs = df['frequency'].values
                        sort_idx = np.argsort(freqs)
                        freqs = freqs[sort_idx]
                        
                        # Convert complex string to complex number and calculate power
                        r_complex = df['r'].apply(complex).values[sort_idx]
                        t_complex = df['t'].apply(complex).values[sort_idx]
                        refls = np.abs(r_complex)**2
                        trans = np.abs(t_complex)**2
                        config_refls.append(refls)
                        config_trans.append(trans)
                    
                    # If we found any curves, compute their average
                    if config_refls and config_trans:
                        avg_refl = np.mean(config_refls, axis=0)
                        avg_trans = np.mean(config_trans, axis=0)
                        all_avg_refls.append(avg_refl)
                        all_avg_trans.append(avg_trans)
    
    # Compute global ranges from all averaged curves
    refl_max = max(max(curve) for curve in all_avg_refls) if all_avg_refls else 1.0
    refl_min = min(min(curve) for curve in all_avg_refls) if all_avg_refls else 0.0
    trans_max = max(max(curve) for curve in all_avg_trans) if all_avg_trans else 1.0
    trans_min = min(min(curve) for curve in all_avg_trans) if all_avg_trans else 0.0
    
    return refl_min, refl_max, trans_min, trans_max

def process_l_value(m_val, l_val, global_ranges):
    plist = [0, 25e-9, 50e-9, 75e-9, 100e-9]  # meters
    olist = [0, 10, 20, 50, 10000]  # radians
    
    refl_min, refl_max, trans_min, trans_max = global_ranges
    
    # Create three different plots with different y-axis limits based on global ranges
    ylims = [
        (-0.1, 1.1, 'RT'),  # Full range for both R and T
        (max(0, refl_min - 0.01), min(1.0, refl_max + 0.01), 'R'),  # Range for reflectance
        (max(0, trans_min - 0.01), min(1.0, trans_max + 0.01), 'T')  # Range for transmittance
    ]
    
    for ymin, ymax, plot_type in ylims:
        fig, axes = plt.subplots(5, 5, figsize=(20, 20), sharex=True, sharey=True)
        
        for p in range(5):
            for o in range(5):
                folder_name = f"l{l_val}_p{p}_o{o}_m{m_val}"
                folder_path = os.path.join(base_dir, folder_name)
                ax = axes[4 - p, o]
                ax.set_ylim(ymin, ymax)
                
                if not os.path.exists(folder_path):
                    continue
                
                for i in range(0, 10):
                    subfolder = os.path.join(folder_path, f"cdm_input_{i}")
                    if not os.path.exists(subfolder):
                        continue
                    
                    file_path = os.path.join(subfolder, "reflection_transmission_complex.csv")
                    if not os.path.isfile(file_path):
                        continue
                        
                    # Read CSV with complex amplitudes
                    df = pd.read_csv(file_path)
                    freqs = df['frequency'].values
                    sort_idx = np.argsort(freqs)
                    freqs = freqs[sort_idx]
                    
                    # Convert complex string to complex number and calculate power
                    r_complex = df['r'].apply(complex).values[sort_idx]
                    t_complex = df['t'].apply(complex).values[sort_idx]
                    refls = np.abs(r_complex)**2
                    trans = np.abs(t_complex)**2
                    
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
        
        plt.suptitle(f'L={l_val} M={m_val} {plot_type} Plot')
        plt.tight_layout()
        output_file = os.path.join(base_dir, f'l{l_val}_m{m_val}_{plot_type.lower()}_plot.png')
        plt.savefig(output_file)
        plt.close()

def main():
    args = parse_args()
    base_dir = args.base_dir

    # Get all subdirectories and extract l and m values
    l_values = set()
    m_values = set()
    pattern = re.compile(r'l(\d+)_p\d+_o\d+_m(\d+)')
    for dirname in os.listdir(base_dir):
        match = pattern.match(dirname)
        if match:
            l_values.add(int(match.group(1)))
            m_values.add(int(match.group(2)))

    # First compute global ranges from all averaged curves
    print("Computing global ranges from averaged curves...")
    global_ranges = compute_global_ranges()
    print(f"Global ranges (R_min, R_max, T_min, T_max): {global_ranges}")

    # Then process each combination of l and m values using these ranges
    for l_val in sorted(l_values):
        for m_val in sorted(m_values):
            print(f"Processing L={l_val} M={m_val}")
            process_l_value(m_val, l_val, global_ranges)

if __name__ == '__main__':
    main()