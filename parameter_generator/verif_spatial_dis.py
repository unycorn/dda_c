from cdm_params import *
import numpy as np
import matplotlib.pyplot as plt

def doubly_perturb_2x2_unit_cell(x_coords, y_coords, shift):
    """
    Modifies a square lattice to create 2x2 unit cells with coordinates:
    (100, 100), (300,0), (0,300), (200,200) nm
    These positions repeat periodically across the lattice
    """
    # Get the lattice spacing from the input coordinates
    spacing = x_coords[1] - x_coords[0]
    
    # Calculate number of points in each direction
    x_unique = np.unique(x_coords)
    y_unique = np.unique(y_coords)
    nx = len(x_unique)
    ny = len(y_unique)
    
    # Reshape arrays to 2D grid for easier manipulation
    x_grid = x_coords.reshape(ny, nx)
    y_grid = y_coords.reshape(ny, nx)
    
    # Unit cell shifts in nanometers converted to meters
    dx = np.array([0 + shift, 300, 0, 300 - shift]) * 1e-9
    dy = np.array([0 + shift, 0, 300, 300 - shift]) * 1e-9
    
    # For each 2x2 block
    for i in range(0, ny-1, 2):
        for j in range(0, nx-1, 2):
            # Base position for this unit cell
            base_x = x_grid[i,j]
            base_y = y_grid[i,j]
            
            # Apply the shifts relative to the base position
            x_grid[i:i+2, j:j+2] = base_x + dx.reshape(2,2)
            y_grid[i:i+2, j:j+2] = base_y + dy.reshape(2,2)
    
    return x_grid.flatten(), y_grid.flatten()

def singly_perturb_2x2_unit_cell(x_coords, y_coords, shift):
    """
    Modifies a square lattice to create 2x2 unit cells with coordinates:
    (100, 100), (300,0), (0,300), (200,200) nm
    These positions repeat periodically across the lattice
    """
    # Get the lattice spacing from the input coordinates
    spacing = x_coords[1] - x_coords[0]
    
    # Calculate number of points in each direction
    x_unique = np.unique(x_coords)
    y_unique = np.unique(y_coords)
    nx = len(x_unique)
    ny = len(y_unique)
    
    # Reshape arrays to 2D grid for easier manipulation
    x_grid = x_coords.reshape(ny, nx)
    y_grid = y_coords.reshape(ny, nx)
    
    # Unit cell shifts in nanometers converted to meters
    dx = np.array([0, 300, 0, 300 - shift]) * 1e-9
    dy = np.array([0, 0, 300, 300 - shift]) * 1e-9
    
    # For each 2x2 block
    for i in range(0, ny-1, 2):
        for j in range(0, nx-1, 2):
            # Base position for this unit cell
            base_x = x_grid[i,j]
            base_y = y_grid[i,j]
            
            # Apply the shifts relative to the base position
            x_grid[i:i+2, j:j+2] = base_x + dx.reshape(2,2)
            y_grid[i:i+2, j:j+2] = base_y + dy.reshape(2,2)
    
    return x_grid.flatten(), y_grid.flatten()

def run_and_measure(is_double, name, shift):
    script_dir = os.path.dirname(os.path.abspath(__file__))
        
    # Set up basic parameters - using 300nm spacing as base
    physical_size = 15e-6  # 15 µm size
    lattice_spacing = 300e-9  # 300 nm spacing
    
    print(f"Using physical size of {physical_size*1e6:.1f} µm with {lattice_spacing*1e9:.0f} nm spacing")

    # Define output directory
    base_output_dir = os.path.join("~", "dda_c", "csv_inputs")
    os.makedirs(base_output_dir, exist_ok=True)

    # Create square lattice
    x_base, y_base = create_square_lattice(lattice_spacing, physical_size)

    if is_double:
        # Modify the lattice to create 2x2 unit cells
        x_base, y_base = doubly_perturb_2x2_unit_cell(x_base, y_base, shift)
        z_base = np.zeros_like(x_base)
    else:
        # Modify the lattice to create 2x2 unit cells
        x_base, y_base = singly_perturb_2x2_unit_cell(x_base, y_base, shift)
        z_base = np.zeros_like(x_base)
    
    # Load u-shape-ideal parameters
    param_file = os.path.join(script_dir, 'u-shape-ideal-cdm-param.csv')
    
    # Create output folder and file
    output_folder = os.path.join(base_output_dir, name)  # Changed folder name
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "cdm_input_0.csv")
    
    # Set up distributions with no disorder
    distributions = [
        ("delta_x", 0, 0),
        ("delta_y", 0, 0),
        ("theta", 0, 0)
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
        if key not in ['delta_x', 'delta_y']:  # f0 and hw already handled
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
    
    # Visualize the lattice
    plt.figure(figsize=(10, 10))
    plt.scatter(x_base * 1e9, y_base * 1e9, s=50)  # Convert to nm for plotting
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('X position (nm)')
    plt.ylabel('Y position (nm)')
    plt.title('2x2 Unit Cell Pattern')
    
    # Highlight the first unit cell
    mask = (x_base < 400e-9) & (y_base < 400e-9)
    plt.scatter(x_base[mask] * 1e9, y_base[mask] * 1e9, s=100, facecolors='none', 
            edgecolors='r', linewidth=2, label='First unit cell')
    
    # Add annotations for the first unit cell coordinates
    for x, y in zip(x_base[mask] * 1e9, y_base[mask] * 1e9):
        plt.annotate(f'({int(x)},{int(y)})', (x, y), 
                    xytext=(10, 10), textcoords='offset points')
    
    plt.legend()
    plt.savefig(os.path.join(output_folder, "2x2_unit_cell_pattern.png"))
    plt.show()
        
    # Change to the output folder and run solver
    original_dir = os.getcwd()
    os.chdir(output_folder)
    
    # Run the solver command from output_folder with expanded home path
    solver_cmd = f"{os.path.expanduser('~/dda_c/dharper/solver')} . 250e12 350e12 50"
    os.system(solver_cmd)
    
    # Change to cdm_input_0 directory for the sampler
    os.chdir("cdm_input_0")
    
    # Run the analytic sampler from cdm_input_0 with expanded home path
    sampler_cmd = f"python {os.path.expanduser('~/dda_c/sampler/analytic_sampler/ana_sampler.py')} . {physical_size * physical_size}"
    os.system(sampler_cmd)
    
    # Return to original directory
    os.chdir(original_dir)

if __name__ == "__main__":
    run_and_measure(False, f"singly_perturbed_2x2_100nm", 100)
    run_and_measure(True, f"doubly_perturbed_2x2_100nm", 100)
    run_and_measure(True, f"doubly_perturbed_2x2_0nm", 0)