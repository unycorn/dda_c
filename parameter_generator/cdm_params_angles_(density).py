import csv
import numpy as np
import random
import os
import pandas as pd
import matplotlib.pyplot as plt

def create_square_lattice(spacing, physical_size):
    """
    Create a square lattice with given spacing that fits within a physical size.
    The lattice starts at (0,0) and extends in positive directions.
    
    :param spacing: Spacing between points in meters (e.g., 300e-9 for 300nm)
    :param physical_size: Physical size of the lattice in meters (e.g., 15e-6 for 15µm)
    :return: Arrays of x and y coordinates in meters
    :raises ValueError: If physical_size is not significantly larger than spacing
    """
    if physical_size < spacing * 2:
        raise ValueError(f"Physical size ({physical_size*1e9:.1f}nm) must be at least twice the spacing ({spacing*1e9:.1f}nm)")
    
    # Calculate number of points that fit within physical size
    num_points = int(np.floor(physical_size / spacing))
    
    if num_points < 2:
        raise ValueError(f"Physical size too small to fit multiple points with given spacing")
    
    # Create arrays starting from 0
    x = np.arange(num_points) * spacing
    y = np.arange(num_points) * spacing
    
    xx, yy = np.meshgrid(x, y)
    return xx.flatten(), yy.flatten()

def create_triangular_lattice(spacing, physical_size):
    """
    Create a triangular lattice with given spacing that fits within a physical size.
    The lattice starts at (0,0) and extends in positive directions.
    The lattice is created by offsetting alternate rows to create equilateral triangles.
    
    :param spacing: Spacing between points in meters (e.g., 300e-9 for 300nm)
    :param physical_size: Physical size of the lattice in meters (e.g., 15e-6 for 15µm)
    :return: Arrays of x and y coordinates in meters
    :raises ValueError: If physical_size is not significantly larger than spacing
    """
    if physical_size < spacing * 2:
        raise ValueError(f"Physical size ({physical_size*1e9:.1f}nm) must be at least twice the spacing ({spacing*1e9:.1f}nm)")
    
    # Calculate the vertical spacing between rows (height of equilateral triangle)
    y_spacing = spacing * np.sqrt(3) / 2
    
    # Calculate number of points that fit within physical size
    num_points_x = int(np.floor(physical_size / spacing))
    num_points_y = int(np.floor(physical_size / y_spacing))
    
    if num_points_x < 2 or num_points_y < 2:
        raise ValueError(f"Physical size too small to fit multiple points with given spacing")
    
    # Create base coordinates
    x_coords = []
    y_coords = []
    
    # Start at (0,0) and extend in positive directions
    for i in range(num_points_y):
        y = i * y_spacing
        row_offset = (spacing / 2) if i % 2 else 0
        
        for j in range(num_points_x):
            x = j * spacing + row_offset
            x_coords.append(x)
            y_coords.append(y)
    
    return np.array(x_coords), np.array(y_coords)

def generate_normal_values(distributions, num_values):
    """
    Generate values from normal distributions.

    :param distributions: List of tuples (name, mean, std_dev) for each distribution.
    :param num_values: Number of values to generate for each distribution.
    :return: Dictionary containing generated values for each distribution.
    """
    return {name: [random.gauss(mean, std_dev) for _ in range(num_values)] for name, mean, std_dev in distributions}

def write_output_csv(filename, data, headers):
    """
    Write data to a CSV file.
    
    :param filename: Output filename
    :param data: Dictionary containing the data
    :param headers: List of column names in desired order
    """
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        
        num_rows = len(data[headers[0]])
        for i in range(num_rows):
            row = [data[header][i] for header in headers]
            writer.writerow(row)

def load_resonator_parameters(csv_file):
    """
    Load resonator parameters from a CSV file.
    
    :param csv_file: Path to the CSV file containing the parameters
    :return: List of tuples (name, mean, std_dev) for each parameter
    """
    df = pd.read_csv(csv_file)
    param_list = []
    
    # First three parameters are always the same
    param_list.extend([
        ("delta_x", 0, 0),  # Will be set later
        ("delta_y", 0, 0),  # Will be set later
        ("theta", 0, 0),    # Will be set later
        ("f0", 0, 0),      # Will be set later
        ("hw", 0, 0),      # Will be set later
    ])
    
    # Add parameters from CSV
    for _, row in df.iterrows():
        param_list.append((row['parameter'], row['mean'], row['std_dev']))
    
    return param_list

if __name__ == "__main__":
    # First load Ammann-Beenker data to determine physical dimensions
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ab_file_path = os.path.join(script_dir, "Iteration4_300nm_density.npy")
    ab_xy_data = np.load(ab_file_path)
    ab_x = np.array([xyi.real for xyi in ab_xy_data])
    ab_y = np.array([xyi.imag for xyi in ab_xy_data])
    
    # Calculate physical size from Ammann-Beenker data
    physical_size = max(
        ab_x.max() - ab_x.min(),
        ab_y.max() - ab_y.min()
    )
    lattice_spacing = 300e-9  # 300 nm spacing
    
    print("area m^2", physical_size**2)
    input()
    print(f"Using physical size of {physical_size*1e6:.1f} µm based on Ammann-Beenker tiling")

    # Define base directory and check if it exists
    base_output_dir = os.path.join(script_dir, "..", "csv_inputs")

    seed_count = 1  # Number of seeds to generate

    # Dictionary of available resonator types and their parameter files
    resonator_files = {
        'c-shape-ideal': 'c-shape-ideal-cdm-param.csv'
    }

    # Dictionary to store all lattice types
    lattices = {
        0: ('square0', create_square_lattice(lattice_spacing, physical_size)),
        1: ('square1', create_square_lattice(0.8 * lattice_spacing, physical_size)),
        2: ('square2', create_square_lattice(0.6 * lattice_spacing, physical_size)),
        3: ('square3', create_square_lattice(0.4 * lattice_spacing, physical_size)),
        4: ('square4', create_square_lattice(0.2 * lattice_spacing, physical_size)),
    }

    M_OFFSET = 0

    # Disorder parameters
    spatial_disorder_degrees = [0]  # meters
    orientational_disorder_degrees = [*np.deg2rad(np.arange(0, 100, 2)), 1e6]

    # Iterate through resonator types, spatial and orientational disorder, and lattice types
    for m, (resonator_type, resonator_filename) in enumerate(resonator_files.items()):
        print(f"\nProcessing resonator type {m}: {resonator_type}")
        param_file = os.path.join(os.path.dirname(__file__), resonator_filename)
        
        for l, (lattice_name, (x_base, y_base)) in lattices.items():
            print(f"Processing lattice type {l}: {lattice_name}")
            z_base = np.zeros_like(x_base)  # z coordinates are all 0
            
            for s, spatial_disorder in enumerate(spatial_disorder_degrees):
                for o, orientational_disorder in enumerate(orientational_disorder_degrees):
                    output_folder = os.path.join(base_output_dir, f"l{l}_p{s}_o{o}_m{m+M_OFFSET}")
                    print(output_folder)
                    os.makedirs(output_folder, exist_ok=True)
                    
                    for i in range(seed_count):
                        output_file = os.path.join(output_folder, f"cdm_input_{i}.csv")
                        
                        # Load parameters from CSV
                        distributions = [
                            ("delta_x", 0, spatial_disorder),
                            ("delta_y", 0, spatial_disorder),
                            ("theta", 0, orientational_disorder)
                        ]
                        
                        # Load all other parameters from the CSV file
                        df = pd.read_csv(param_file)
                        for _, row in df.iterrows():
                            distributions.append((row['parameter'], row['mean'], row['std_dev']))

                        # Generate disorder parameters directly
                        num_points = len(x_base)
                        disorder_data = generate_normal_values(distributions, num_points)
                        
                        # Create final data dictionary
                        data = {}
                        data['x'] = x_base + disorder_data['delta_x'][:num_points]
                        data['y'] = y_base + disorder_data['delta_y'][:num_points]
                        data['z'] = z_base
                        
                        # Create the eight resonator parameter columns from the loaded distributions
                        for prefix in ['ee', 'em', 'me', 'mm']:
                            data[f'{prefix}_f0'] = disorder_data['f0'][:num_points]
                            data[f'{prefix}_hw'] = disorder_data['hw'][:num_points]
                        
                        # Copy all other parameters
                        for key in disorder_data:
                            if key not in ['delta_x', 'delta_y']:  # f0 and hw are already handled
                                data[key] = disorder_data[key][:num_points]
                        
                        # Use explicit column order
                        headers = ['x', 'y', 'z', 'theta',
                                'ee_f0', 'ee_hw', 'ee_A', 'ee_B', 'ee_C',
                                'em_f0', 'em_hw', 'em_A', 'em_B', 'em_C',
                                'me_f0', 'me_hw', 'me_A', 'me_B', 'me_C',
                                'mm_f0', 'mm_hw', 'mm_A', 'mm_B', 'mm_C']
                        
                        # Write the output file
                        write_output_csv(output_file, data, headers)
                        print(f"Generated CDM input parameters saved to {output_file}")
                    
                    
