import numpy as np
import matplotlib.pyplot as plt
import os

def gaussian_2d(x, y, x0, y0, sigma):
    """2D Gaussian function centered at (x0, y0) with standard deviation sigma"""
    # For zero displacement, use a very narrow Gaussian
    if sigma == 0:
        sigma = 1e-12  # Very small but non-zero value
    # Properly normalized 2D Gaussian
    norm_factor = 1 / (2 * np.pi * sigma**2)
    return norm_factor * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

def generate_heatmap(std_dev):
    # Create a 5x5 grid with 300nm spacing
    spacing = 300e-9  # 300 nanometers in meters
    n_points = 5

    # If std_dev is 0, use 1nm instead
    if std_dev == 0:
        std_dev = 1e-9

    # Create base grid points
    x_grid = np.arange(n_points) * spacing
    y_grid = np.arange(n_points) * spacing

    # Create high-resolution grid for smooth plotting
    plot_margin = max(2 * std_dev, spacing/4)
    x = np.linspace(x_grid[0] - plot_margin, x_grid[-1] + plot_margin, 500)
    y = np.linspace(y_grid[0] - plot_margin, y_grid[-1] + plot_margin, 500)
    X, Y = np.meshgrid(x, y)

    # Calculate probability distribution (sum of Gaussians)
    P = np.zeros_like(X)
    for x0 in x_grid:
        for y0 in y_grid:
            P += gaussian_2d(X, Y, x0, y0, std_dev)

    # Convert to microns for plotting
    X_microns = X * 1e6
    Y_microns = Y * 1e6
    x_grid_microns = x_grid * 1e6
    y_grid_microns = y_grid * 1e6

    # Create the figure
    plt.figure(figsize=(8, 8))

    # Plot the probability distribution using imshow
    plt.imshow(P, extent=[X_microns.min(), X_microns.max(), 
                         Y_microns.min(), Y_microns.max()],
              origin='lower', cmap='Blues', aspect='equal')

    # Add grid aligned with original point positions
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(x_grid_microns)
    plt.yticks(y_grid_microns)

    # Set labels with units
    plt.xlabel('X (µm)')
    plt.ylabel('Y (µm)')

    # Add title showing parameters
    if std_dev == 1e-9:  # Was originally 0
        plt.title('Position Distribution\n(σ = 1nm)')
    else:
        plt.title(f'Position Distribution\n(σ = {std_dev*1e9:.0f}nm)')

    # Create directory if it doesn't exist
    os.makedirs(os.path.join('analysis', 'heatmap_smooth_figure'), exist_ok=True)

    # Save the figure
    filename = f'heatmap_std_{std_dev*1e9:.0f}nm.png'
    save_path = os.path.join('analysis', 'heatmap_smooth_figure', filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Generate figures for each standard deviation
std_devs = [0, 25e-9, 50e-9, 75e-9, 100e-9]
for std_dev in std_devs:
    generate_heatmap(std_dev)