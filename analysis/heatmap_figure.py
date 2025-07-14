import numpy as np
import matplotlib.pyplot as plt
import os

def generate_heatmap(std_dev):
    # Create a 4x4 grid with 300nm spacing
    spacing = 300e-9  # 300 nanometers in meters
    n_points = 4
    n_realizations = 500  # Number of overlaid point sets

    # Create coordinate arrays
    x = np.arange(n_points) * spacing
    y = np.arange(n_points) * spacing

    # Create meshgrid
    X, Y = np.meshgrid(x, y)

    # Convert to microns for plotting
    X_microns = X * 1e6
    Y_microns = Y * 1e6

    # Create the figure
    plt.figure(figsize=(8, 8))

    # Add grid aligned with original point positions
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(X_microns[0, :])
    plt.yticks(Y_microns[:, 0])

    # Plot original positions as light gray points
    plt.scatter(X_microns.flatten(), Y_microns.flatten(), color='lightgray', 
               s=150, alpha=0.5, label='Original')

    # Add multiple sets of displaced points
    for _ in range(n_realizations):
        # Generate new displacements for each set
        dx = np.random.normal(0, std_dev, X.shape)
        dy = np.random.normal(0, std_dev, Y.shape)
        
        # Add displacements to coordinates
        X_displaced = X + dx
        Y_displaced = Y + dy
        
        # Convert to microns
        X_displaced_microns = X_displaced * 1e6
        Y_displaced_microns = Y_displaced * 1e6
        
        # Plot displaced points with transparency
        plt.scatter(X_displaced_microns.flatten(), Y_displaced_microns.flatten(), 
                   color='blue', s=150, alpha=0.04)

    # Set labels with units
    plt.xlabel('X (µm)')
    plt.ylabel('Y (µm)')

    # Make the plot square and equal aspect ratio
    plt.axis('square')

    # Add title showing parameters
    if std_dev == 0:
        plt.title('Position Distribution\n(No displacement)')
    else:
        plt.title(f'Position Distribution\n(σ = {std_dev*1e9:.0f}nm, N = {n_realizations} realizations)')

    # Save the figure
    filename = f'heatmap_std_{std_dev*1e9:.0f}nm.png'
    save_path = os.path.join('analysis', 'heatmap_figure', filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Generate figures for each standard deviation
std_devs = [0, 25e-9, 50e-9, 75e-9, 100e-9]
for std_dev in std_devs:
    generate_heatmap(std_dev)