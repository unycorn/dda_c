#!/usr/bin/env python3
"""
Script to iterate through folders and load power.npy files from each subfolder.

Usage:
    python load_power_data.py /path/to/parent/folder
"""

import os
import sys
import numpy as np
from pathlib import Path

def load_power_data_from_folders(parent_folder_path):
    """
    Load power.npy files from subdirectories in the given parent folder.
    Each power.npy file contains [freq_list, P0_list, Pt_list].
    
    Args:
        parent_folder_path (str): Path to the parent directory containing subdirectories
        
    Returns:
        dict: Dictionary with combined data structure:
              {
                'frequencies': combined frequency array,
                'data': {
                  'folder_name1': {'P0': array, 'Pt': array},
                  'folder_name2': {'P0': array, 'Pt': array},
                  ...
                },
                'folder_names': list of folder names
              }
    """
    parent_path = Path(parent_folder_path)
    
    if not parent_path.exists():
        raise FileNotFoundError(f"Parent folder does not exist: {parent_folder_path}")
    
    if not parent_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {parent_folder_path}")
    
    # Get all subdirectories in the parent folder
    subdirectories = [d for d in parent_path.iterdir() if d.is_dir()]
    
    if not subdirectories:
        print(f"No subdirectories found in {parent_folder_path}")
        return {'frequencies': None, 'data': {}, 'folder_names': []}
    
    print(f"Found {len(subdirectories)} subdirectories in {parent_folder_path}")
    
    combined_data = {'frequencies': None, 'data': {}, 'folder_names': []}
    successful_loads = 0
    failed_loads = 0
    
    for folder in subdirectories:
        folder_name = folder.name
        power_file_path = folder / "cdm_input_0" / "power.npy"
        
        try:
            if power_file_path.exists():
                # Load the data: [freq_list, P0_list, Pt_list]
                raw_data = np.load(power_file_path)
                
                if raw_data.shape[0] != 3:
                    print(f"✗ Unexpected data format in {folder_name}: expected shape (3, N), got {raw_data.shape}")
                    failed_loads += 1
                    continue
                
                freq_list = raw_data[0]
                P0_list = raw_data[1]
                Pt_list = raw_data[2]
                
                # Store the data for this folder
                combined_data['data'][folder_name] = {
                    'P0': P0_list,
                    'Pt': Pt_list
                }
                combined_data['folder_names'].append(folder_name)
                
                # Use the first frequency array as the reference (assuming all are the same)
                if combined_data['frequencies'] is None:
                    combined_data['frequencies'] = freq_list
                elif not np.allclose(combined_data['frequencies'], freq_list):
                    print(f"⚠ Warning: Frequency array in {folder_name} differs from reference")
                
                successful_loads += 1
                print(f"✓ Loaded {folder_name}: {len(freq_list)} frequency points")
                
            else:
                print(f"✗ Missing file: {power_file_path}")
                failed_loads += 1
                
        except Exception as e:
            print(f"✗ Error loading {power_file_path}: {e}")
            failed_loads += 1
    
    print(f"\nSummary:")
    print(f"  Successfully loaded: {successful_loads}")
    print(f"  Failed to load: {failed_loads}")
    print(f"  Total folders processed: {len(subdirectories)}")
    
    # Sort folder names for consistent ordering
    combined_data['folder_names'].sort()
    
    return combined_data

def main():
    """Main function to handle command line arguments and execute the loading."""
    if len(sys.argv) != 2:
        print("Usage: python load_power_data.py <folder_path>")
        print("Example: python load_power_data.py /path/to/csv_inputs")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    try:
        combined_data = load_power_data_from_folders(folder_path)
        
        if combined_data['data']:
            print(f"\nSuccessfully combined power data from {len(combined_data['data'])} folders:")
            
            # Display frequency information
            if combined_data['frequencies'] is not None:
                freq_array = combined_data['frequencies']
                print(f"\nFrequency range: {freq_array[0]*1e-12:.2f} - {freq_array[-1]*1e-12:.2f} THz")
                print(f"Number of frequency points: {len(freq_array)}")
            
            # Display data for each folder
            print(f"\nData structure:")
            for folder_name in combined_data['folder_names']:
                data = combined_data['data'][folder_name]
                P0_stats = f"P0: mean={np.mean(data['P0']):.2e}, max={np.max(data['P0']):.2e}"
                Pt_stats = f"Pt: mean={np.mean(data['Pt']):.2e}, max={np.max(data['Pt']):.2e}"
                print(f"  {folder_name}: {P0_stats}, {Pt_stats}")
            
            # Save combined data to a single file
            output_file = "combined_power_data.npz"
            np.savez(output_file,
                    frequencies=combined_data['frequencies'],
                    folder_names=combined_data['folder_names'],
                    **{f"{name}_P0": combined_data['data'][name]['P0'] 
                       for name in combined_data['folder_names']},
                    **{f"{name}_Pt": combined_data['data'][name]['Pt'] 
                       for name in combined_data['folder_names']})
            print(f"\nCombined data saved to: {output_file}")
            
            # Example analysis: calculate transmission efficiency
            print(f"\nTransmission efficiency (Pt/P0) for each folder:")
            for folder_name in combined_data['folder_names']:
                data = combined_data['data'][folder_name]
                # Avoid division by zero
                mask = data['P0'] != 0
                if np.any(mask):
                    transmission = np.where(mask, data['Pt'] / data['P0'], 0)
                    mean_transmission = np.mean(transmission[mask])
                    print(f"  {folder_name}: {mean_transmission:.4f}")
                else:
                    print(f"  {folder_name}: N/A (P0 is zero)")
                    
        else:
            print("No power data could be loaded.")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()