import numpy as np
import matplotlib.pyplot as plt
import os

def generate_angle_dist(std_dev_degrees):
    """Generate angular distribution and create circular histogram"""
    # Generate 30 million angles from normal distribution
    n_samples = 30_000_000
    angles = np.random.normal(0, std_dev_degrees, n_samples)
    
    # Convert to radians and wrap to [-pi, pi]
    angles_rad = np.deg2rad(angles) % (2 * np.pi)
    
    # Create figure with polar projection
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    
    # Create histogram
    n_bins = 40  # 9-degree bins
    counts, bin_edges = np.histogram(angles_rad, bins=n_bins, range=(0, 2*np.pi))
    
    # Get bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot histogram as bars on polar axes
    bars = ax.bar(bin_centers, counts, width=2*np.pi/n_bins)
    
    # Customize the plot
    ax.set_theta_zero_location('N')  # 0 degrees at North
    ax.set_theta_direction(-1)  # clockwise
    
    # Remove all tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Add title showing standard deviation
    if std_dev_degrees == 0:
        plt.title('Angular Distribution\n(σ = 0°)')
    else:
        plt.title(f'Angular Distribution\n(σ = {std_dev_degrees:.0f}°)')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.join('analysis', 'angle_figures'), exist_ok=True)
    
    # Save the figure
    filename = f'angle_dist_std_{std_dev_degrees:.0f}deg.png'
    save_path = os.path.join('analysis', 'angle_figures', filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# List of standard deviations to use (in degrees)
std_devs = [
    0,
    np.rad2deg(np.deg2rad(10)),  # 10 degrees
    np.rad2deg(np.deg2rad(20)),  # 20 degrees
    np.rad2deg(np.deg2rad(50)),  # 50 degrees
    np.rad2deg(np.deg2rad(1_000_000))  # 1,000,000 degrees
]

# Generate figures for each standard deviation
for std_dev in std_devs:
    generate_angle_dist(std_dev)