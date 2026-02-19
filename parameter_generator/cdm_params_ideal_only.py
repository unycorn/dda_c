import csv
import numpy as np
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import random
from scipy.stats import truncnorm

def pairwise_correlation_multiple(x_sets, y_sets, title=""):
    """
    x_sets: list of x arrays
    y_sets: list of y arrays
    """

    all_nn_distances = []

    for x, y in zip(x_sets, y_sets):
        points = np.column_stack((x, y))
        tree = cKDTree(points)

        distances, _ = tree.query(points, k=2)
        nn = distances[:, 1] * 1e9  # nm

        all_nn_distances.append(nn)

    all_nn_distances = np.concatenate(all_nn_distances)

    plt.figure()
    plt.hist(all_nn_distances, bins=100, density=False, range=(0, 600))
    plt.xlabel("Nearest-neighbor distance (nm)")
    plt.ylabel("Count")
    plt.title(f"Nearest-Neighbor Distance Distribution {title}")
    plt.savefig(f"PCF_translational_disorder_{title}.pdf")
    plt.show()


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

def generate_lattice_with_min_distance(N, a, sigma, dmin, max_attempts=1000000000000):
    """
    Generate NxN square lattice with Gaussian perturbations,
    rejecting samples that violate minimum distance.

    Returns:
        ideal_positions        : (N^2, 2)
        perturbed_positions    : (N^2, 2)
    """

    ideal = np.empty((N, N, 2))
    perturbed = np.empty((N, N, 2))

    for i in range(N):
        for j in range(N):

            ideal_pos = np.array([i*a, j*a])
            ideal[i, j] = ideal_pos

            for _ in range(max_attempts):

                candidate = ideal_pos + np.random.normal(scale=sigma, size=2)

                ok = True

                # Check neighbors within ±2 lattice sites
                for di in (-2, -1, 0, 1, 2):
                    for dj in (-2, -1, 0, 1, 2):

                        ni = i + di
                        nj = j + dj

                        if 0 <= ni < N and 0 <= nj < N:

                            # Only check already placed sites
                            if ni < i or (ni == i and nj < j):

                                dist = np.linalg.norm(candidate - perturbed[ni, nj])
                                if dist < dmin:
                                    ok = False
                                    break
                    if not ok:
                        break

                if ok:
                    perturbed[i, j] = candidate
                    break

            else:
                raise RuntimeError(f"Failed to place site ({i},{j}).")

    return perturbed.reshape(-1, 2)

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

def generate_normal_values(distributions, num_values, truncation=None):
    """
    Generate values from normal distributions.

    :param distributions: List of tuples (name, mean, std_dev) for each distribution.
    :param num_values: Number of values to generate for each distribution.
    :param truncation: Optional dict mapping name -> (lower, upper) bounds.
                       If provided and name matches, values are drawn from a
                       properly normalized truncated Gaussian.
    :return: Dictionary containing generated values for each distribution.
    """

    results = {}

    for name, mean, std_dev in distributions:

        if std_dev == 0:
            values = [mean] * num_values

        elif truncation and name in truncation:
            lower, upper = truncation[name]

            a = (lower - mean) / std_dev
            b = (upper - mean) / std_dev

            dist = truncnorm(a, b, loc=mean, scale=std_dev)
            values = dist.rvs(num_values)

        else:
            values = [random.gauss(mean, std_dev) for _ in range(num_values)]

        results[name] = values

    return results


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

    physical_size = 30e-6
    lattice_spacing = 300e-9  # 300 nm spacing
    
    print("area m^2", physical_size**2)
    # input()
    print(f"Using physical size of {physical_size*1e6:.1f} µm based on Ammann-Beenker tiling")

    # Define base directory and check if it exists
    base_output_dir = os.path.join(script_dir, "..", "csv_inputs")

    seed_count = 1  # Number of seeds to generate

    # Dictionary of available resonator types and their parameter files
    resonator_files = {
        # 'c-shape-ideal': 'c-shape-ideal-cdm-param_NoRadLoss.csv',
        # 'c-shape-36': 'c-shape-36-cdm-param_NoRadLoss.csv',
        # 'c-shape-28': 'c-shape-28-cdm-param_NoRadLoss.csv',
        'u-shape-ideal': 'u-shape-ideal-cdm-param_NoRadLoss.csv',
        # 'u-shape-37': 'u-shape-37-cdm-param_NoRadLoss.csv',
        # 'u-shape-29': 'u-shape-29-cdm-param_NoRadLoss.csv'
    }

    # Dictionary to store all lattice types
    lattices = {
        0: ('square', create_square_lattice(lattice_spacing, physical_size)),
        # 1: ('triangular', create_triangular_lattice(lattice_spacing/np.sqrt(np.sin(np.pi / 3)), physical_size)),
        # 2: ('ammann-beenker', (ab_x, ab_y))
    }

    M_OFFSET = 0

    # Disorder parameters
    spatial_disorder_degrees = [0]  # meters
    orientational_disorder_degrees = [0]  # radians

    all_x = [[],[],[],[],[]]
    all_y = [[],[],[],[],[]]

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
                        disorder_data = generate_normal_values(distributions, num_points)#, truncation={"delta_x": (-100e-9, 100e-9),"delta_y": (-100e-9, 100e-9)})
                        
                        if 'ideal' not in resonator_type:
                            disorder_data['ee_A'] = np.abs(np.array(disorder_data['ee_A']))
                            disorder_data['mm_A'] = np.abs(np.array(disorder_data['mm_A']))

                            off_diagonals = np.sqrt(disorder_data['ee_A']*disorder_data['mm_A'])
                            disorder_data['em_A'] = off_diagonals
                            disorder_data['me_A'] = off_diagonals

                        points_with_min_dist = generate_lattice_with_min_distance(1, 300e-9, spatial_disorder, 100e-9)

                        # Create final data dictionary
                        data = {}
                        data['x'] = points_with_min_dist[:,0] #x_base + disorder_data['delta_x'][:num_points]
                        data['y'] = points_with_min_dist[:,1] #y_base + disorder_data['delta_y'][:num_points]
                        data['z'] = z_base


                        all_x[s].append(data['x'])
                        all_y[s].append(data['y'])

                        
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

    for s in range(1,5):
        print(s)
        if s > 0:
            pairwise_correlation_multiple(all_x[s], all_y[s], title=s)