from cdm_params import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_and_tile_positions(csv_file, supercell_size=15):
    """
    Loads positions from a CSV file and tiles them in a supercell array.
    
    Parameters:
    csv_file: Path to CSV file with columns x,y,z,rotation_deg
    supercell_size: Number of unit cells in each direction (default 15x15)
    
    Returns:
    x_coords, y_coords, z_coords, rotations: Arrays of tiled positions and rotations
    """
    # Load the unit cell data
    df = pd.read_csv(csv_file)
    
    # Extract positions and rotations
    unit_x = df['x'].values * 1e-9  # Convert nm to meters
    unit_y = df['y'].values * 1e-9  # Convert nm to meters
    unit_z = df['z'].values * 1e-9  # Convert nm to meters
    unit_rot = df['rotation_deg'].values  # Rotations in degrees
    
    # Determine unit cell size from the data
    unit_cell_size_x = np.max(unit_x) - np.min(unit_x) + 300e-9  # Add spacing
    unit_cell_size_y = np.max(unit_y) - np.min(unit_y) + 300e-9  # Add spacing
    
    print(f"Unit cell size: {unit_cell_size_x*1e9:.1f} nm x {unit_cell_size_y*1e9:.1f} nm")
    print(f"Creating {supercell_size}x{supercell_size} supercell")
    
    # Initialize arrays for the full supercell
    total_points = len(unit_x) * supercell_size * supercell_size
    x_coords = np.zeros(total_points)
    y_coords = np.zeros(total_points)
    z_coords = np.zeros(total_points)
    rotations = np.zeros(total_points)
    
    # Tile the unit cell
    idx = 0
    for i in range(supercell_size):
        for j in range(supercell_size):
            # Calculate offset for this tile
            offset_x = i * unit_cell_size_x
            offset_y = j * unit_cell_size_y
            
            # Copy unit cell with offset
            start_idx = idx
            end_idx = idx + len(unit_x)
            
            x_coords[start_idx:end_idx] = unit_x + offset_x
            y_coords[start_idx:end_idx] = unit_y + offset_y
            z_coords[start_idx:end_idx] = unit_z
            rotations[start_idx:end_idx] = unit_rot
            
            idx = end_idx
    
    return x_coords, y_coords, z_coords, rotations

def run_and_measure(csv_file, name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Loading positions from {csv_file}")

    # Define output directory
    base_output_dir = os.path.expanduser(os.path.join("~", "dda_c", "csv_inputs"))
    os.makedirs(base_output_dir, exist_ok=True)

    # Load and tile positions from CSV
    x_base, y_base, z_base, rotations = load_and_tile_positions(csv_file, supercell_size=15)
    
    # Calculate physical size from the actual coordinates
    physical_size_x = np.max(x_base) - np.min(x_base)
    physical_size_y = np.max(y_base) - np.min(y_base)
    physical_size = max(physical_size_x, physical_size_y)
    
    print(f"Physical size: {physical_size*1e6:.1f} µm x {physical_size*1e6:.1f} µm")
    
    # Load u-shape-ideal parameters
    param_file = os.path.join(script_dir, 'u-shape-ideal-cdm-param.csv')
    
    # Create output folder and file
    output_folder = os.path.join(base_output_dir, name)
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "cdm_input_0.csv")
    
    # Set up distributions with no positional disorder (positions come from CSV)
    distributions = [
        ("delta_x", 0, 0),
        ("delta_y", 0, 0),
        ("theta", 0, 0)  # Will be overridden with CSV rotations
    ]
    
    # Load parameters from CSV file
    df = pd.read_csv(param_file)
    for _, row in df.iterrows():
        distributions.append((row['parameter'], row['mean'], row['std_dev']))

    # Generate parameters
    num_points = len(x_base)
    disorder_data = generate_normal_values(distributions, num_points)
    
    # Create final data dictionary
    data = {}
    data['x'] = x_base
    data['y'] = y_base
    data['z'] = z_base
    
    # Create the resonator parameter columns
    for prefix in ['ee', 'em', 'me', 'mm']:
        data[f'{prefix}_f0'] = disorder_data['f0'][:num_points]
        data[f'{prefix}_hw'] = disorder_data['hw'][:num_points]
    
    # Copy other parameters
    for key in disorder_data:
        if key not in ['delta_x', 'delta_y', 'theta']:  # These are handled separately
            data[key] = disorder_data[key][:num_points]
    
    # Use rotations from CSV file (convert degrees to radians)
    data['theta'] = np.deg2rad(rotations)
    
    # Use explicit column order
    headers = ['x', 'y', 'z', 'theta',
            'ee_f0', 'ee_hw', 'ee_A', 'ee_B', 'ee_C',
            'em_f0', 'em_hw', 'em_A', 'em_B', 'em_C',
            'me_f0', 'me_hw', 'me_A', 'me_B', 'me_C',
            'mm_f0', 'mm_hw', 'mm_A', 'mm_B', 'mm_C']
    
    # Write the output file
    write_output_csv(output_file, data, headers)
    print(f"Generated CDM input parameters saved to {output_file}")
    
    # Visualize the lattice
    plt.figure(figsize=(12, 12))
    plt.scatter(x_base * 1e9, y_base * 1e9, s=20, alpha=0.6)  # Convert to nm for plotting
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlabel('X position (nm)')
    plt.ylabel('Y position (nm)')
    plt.title(f'Tiled Positions from CSV ({name})')
    
    # Highlight the first unit cell
    unit_size_nm = 1000  # Approximate unit cell size in nm
    mask = (x_base * 1e9 < unit_size_nm) & (y_base * 1e9 < unit_size_nm)
    plt.scatter(x_base[mask] * 1e9, y_base[mask] * 1e9, s=100, facecolors='none', 
            edgecolors='r', linewidth=2, label='First unit cell')
    
    plt.legend()
    plt.savefig(os.path.join(output_folder, "tiled_positions_pattern.png"), dpi=150)
    plt.show()
    
    print(f"Total number of particles: {num_points}")
        
    # Change to the output folder and run solver
    original_dir = os.getcwd()
    os.chdir(output_folder)
    
    # Run the solver command from output_folder with expanded home path
    solver_cmd = f"{os.path.expanduser('~/dda_c/solver')} . 250e12 350e12 50"
    os.system(solver_cmd)
    
    # Change to cdm_input_0 directory for the sampler
    os.chdir(os.path.expanduser(os.path.join(output_folder, "cdm_input_0")))
    
    # Run the analytic sampler from cdm_input_0 with expanded home path
    sampler_cmd = f"python {os.path.expanduser('~/dda_c/sampler/analytic_sampler/ana_sampler.py')} . {physical_size * physical_size}"
    os.system(sampler_cmd)
    
    # Return to original directory
    os.chdir(original_dir)

if __name__ == "__main__":
    # Example usage - you'll need to provide the path to your CSV file
    csv_file_path = "0_bl_lattice_4by4.csv"  # Update this path to your actual CSV file
    
    # Check if CSV file exists
    if os.path.exists(csv_file_path):
        run_and_measure(csv_file_path, "tiled_from_csv")
    else:
        print(f"CSV file not found: {csv_file_path}")
        print("Please create a CSV file with columns: x,y,z,rotation_deg")
        print("Example content:")
        print("x,y,z,rotation_deg")
        print("42.925,-113.192,0.0,-6.096")
        print("5.219,417.103,0.0,43.847")
        print("...")