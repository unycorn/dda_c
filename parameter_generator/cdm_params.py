import csv
import numpy as np
import random
import os
import pandas as pd
import matplotlib.pyplot as plt

def create_square_lattice(spacing, size):
    """
    Create a square lattice with given spacing, starting from origin (0,0).
    
    :param spacing: Spacing between points in meters
    :param size: Number of points on each side
    :return: Arrays of x and y coordinates
    """
    x = np.arange(size) * spacing  # From 0 to (size-1)*spacing
    y = np.arange(size) * spacing
    xx, yy = np.meshgrid(x, y)
    return xx.flatten(), yy.flatten()

def create_triangular_lattice(spacing, size):
    """
    Create a triangular lattice with given spacing, starting from origin (0,0).
    The lattice is created by offsetting alternate rows to create equilateral triangles.
    
    :param spacing: Spacing between points in meters
    :param size: Number of points on each side
    :return: Arrays of x and y coordinates
    """
    # Calculate the vertical spacing between rows (height of equilateral triangle)
    y_spacing = spacing * np.sqrt(3) / 2
    
    # Create base coordinates
    x_coords = []
    y_coords = []
    
    for i in range(size):
        for j in range(size):
            # For odd rows, offset x-coordinate by half the spacing
            x = j * spacing + (i % 2) * (spacing / 2)
            y = i * y_spacing
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
    # Parameters
    lattice_spacing = 300e-9  # 300 nm spacing
    lattice_size = 50        # 50x50 lattice

    # Define base directory and check if it exists
    base_dir = "/home/dharper/dda_c"
    base_output_dir = os.path.join(base_dir, "csv_inputs")
    # if os.path.exists(base_output_dir):
    #     print(f"Error: Output directory {base_output_dir} already exists. Please remove it first.")
    #     exit(1)

    seed_count = 20  # Number of seeds to generate

    # Dictionary of available resonator types and their parameter files
    # resonator_files = {
    #     'c-shape-36': 'c-shape-36-cdm-param.csv',
    #     'c-shape-28': 'c-shape-28-cdm-param.csv',
    #     'u-shape-37': 'u-shape-37-cdm-param.csv',
    #     'u-shape-29': 'u-shape-29-cdm-param.csv'
    # }
    resonator_files = {
        'c-shape-ideal': 'c-shape-ideal-cdm-param.csv',
        'u-shape-ideal': 'u-shape-ideal-cdm-param.csv'
    }

    # Select lattice type ('square' or 'triangular')
    lattice_type = 'square'  # Change this to use different lattice types
    M_OFFSET = 2

    # Create base lattice
    if lattice_type == 'square':
        x_base, y_base = create_square_lattice(lattice_spacing, lattice_size)
    else:
        x_base, y_base = create_triangular_lattice(lattice_spacing, lattice_size)
    
    # Visualize the lattice
    plt.figure(figsize=(10, 10))
    plt.scatter(x_base, y_base, c='blue', s=10)
    plt.title(f'Generated {lattice_type.capitalize()} Lattice')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')  # Make sure the aspect ratio is 1:1
    plt.grid(True)
    plt.show()
    
    z_base = np.zeros_like(x_base)  # z coordinates are all 0

    # Disorder parameters
    spatial_disorder_degrees = [0, 25e-9, 50e-9, 75e-9, 100e-9]  # meters
    orientational_disorder_degrees = [0, np.deg2rad(10), np.deg2rad(20), np.deg2rad(50), np.deg2rad(1_000_000)]  # radians

    # Iterate through resonator types, spatial and orientational disorder
    for m, (resonator_type, resonator_filename) in enumerate(resonator_files.items()):
        print(f"\nProcessing resonator type {m}: {resonator_type}")
        param_file = os.path.join(os.path.dirname(__file__), resonator_filename)
        
        for s, spatial_disorder in enumerate(spatial_disorder_degrees):
            for o, orientational_disorder in enumerate(orientational_disorder_degrees):
                output_folder = os.path.join(base_output_dir, f"p{s}_o{o}_m{m+M_OFFSET}")
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