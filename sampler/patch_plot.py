#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import pandas as pd
import argparse
import glob
import os
from matplotlib.animation import FuncAnimation, FFMpegWriter

def read_polarizations_binary(filename):
    with open(filename, 'rb') as f:
        # Read N (4-byte integer)
        N = np.fromfile(f, dtype=np.int32, count=1)[0]
        
        # Read frequency (8-byte double)
        freq = np.fromfile(f, dtype=np.float64, count=1)[0]
        
        # Read complex doubles (2 components per point: Ex and Mz)
        data = np.fromfile(f, dtype=np.complex128, count=2*N)
        
        return N, freq, data.reshape(-1, 2)  # reshape to (N, 2) array

def voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct infinite voronoi regions in a 2D diagram to finite regions."""
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        
        if all(v >= 0 for v in vertices):
            # Already finite region
            new_regions.append(vertices)
            continue

        # Reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # Sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

class VoronoiAnimation:
    def __init__(self, positions, thetas, pols_files):
        self.positions = positions
        self.thetas = thetas
        self.pols_files = pols_files
        self.vor = Voronoi(positions[:, :2])
        self.regions, self.vertices = voronoi_finite_polygons_2d(self.vor)
        
        # Calculate arrow components
        self.U = np.cos(thetas)
        self.V = np.sin(thetas)
        
        # Setup the figure
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # Convert positions to microns for display
        positions_microns = positions * 1e6  # assuming input is in meters
        
        # Set plot limits
        margin = 0.1
        x_min, x_max = positions_microns[:, 0].min(), positions_microns[:, 0].max()
        y_min, y_max = positions_microns[:, 1].min(), positions_microns[:, 1].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        self.ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
        self.ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
        
        # Set plot properties
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (µm)')
        self.ax.set_ylabel('Y (µm)')
        
        # Initialize colorbar
        self.smap = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        self.cbar = self.fig.colorbar(self.smap, ax=self.ax, 
                                    label='|px| (C⋅m)')  # dipole moment units
        
        # Title for frequency display
        self.title = self.ax.set_title('')

    def init_animation(self):
        self.ax.cla()
        return []

    def animate(self, i):
        self.ax.cla()
        
        # Read polarization data for this frame
        N, freq, polarizations = read_polarizations_binary(self.pols_files[i])
        
        # Calculate colors
        colors = np.abs(polarizations[:, 0])  # px component
        colors = (colors - colors.min()) / (colors.max() - colors.min())
        
        # Update colorbar
        self.smap.set_array(colors)
        
        # Convert positions to microns for display
        positions_microns = self.positions * 1e6  # assuming input is in meters
        
        # Plot polygons
        for region, color in zip(self.regions, colors):
            polygon = self.vertices[region] * 1e6  # convert vertices to microns
            self.ax.fill(*zip(*polygon), alpha=0.6, c=plt.cm.viridis(color))
        
        # Plot arrows
        self.ax.quiver(positions_microns[:, 0], positions_microns[:, 1], 
                      self.U, self.V, colors, cmap='viridis',
                      scale=50, width=0.002, headwidth=4,
                      headlength=5, headaxislength=4.5,
                      pivot='middle')
        
        # Update title with frequency
        freq_thz = freq / 1e12
        self.ax.set_title(f'Frequency: {freq_thz:.2f} THz')
        
        # Maintain plot properties
        margin = 0.01
        x_min, x_max = positions_microns[:, 0].min(), positions_microns[:, 0].max()
        y_min, y_max = positions_microns[:, 1].min(), positions_microns[:, 1].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        self.ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
        self.ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (µm)')
        self.ax.set_ylabel('Y (µm)')
        
        return []

def create_animation(positions, thetas, pols_folder, output_file):
    # Get all .pols files and sort them by frequency
    pols_files = glob.glob(os.path.join(pols_folder, '*.pols'))
    
    # Read frequencies to sort files
    freq_file_pairs = []
    for file in pols_files:
        with open(file, 'rb') as f:
            N = np.fromfile(f, dtype=np.int32, count=1)[0]
            freq = np.fromfile(f, dtype=np.float64, count=1)[0]
            freq_file_pairs.append((freq, file))
    
    # Sort by frequency
    freq_file_pairs.sort()
    sorted_pols_files = [pair[1] for pair in freq_file_pairs]
    
    # Create animation
    anim = VoronoiAnimation(positions, thetas, sorted_pols_files)
    animation = FuncAnimation(anim.fig, anim.animate,
                            frames=len(sorted_pols_files),
                            init_func=anim.init_animation,
                            interval=200,  # 200ms between frames
                            blit=True)
    
    # Save animation
    writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'),
                         bitrate=2000)
    animation.save(output_file, writer=writer)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Create animated Voronoi diagram of polarization data')
    parser.add_argument('csv_file', help='CSV file with x,y,z,theta columns')
    parser.add_argument('-o', '--output', 
                       help='Output video file (default: same name as input CSV but with .mp4 extension)')
    
    args = parser.parse_args()
    
    # Read position data
    df = pd.read_csv(args.csv_file)
    positions = df[['x', 'y', 'z']].values
    thetas = df['theta'].values
    
    # Get the pols folder path by removing .csv from the input file path
    pols_folder = os.path.splitext(args.csv_file)[0]
    if not os.path.isdir(pols_folder):
        raise ValueError(f"Could not find polarization data folder: {pols_folder}")
    
    # Set default output filename to be the same as input but with .mp4 extension
    if args.output is None:
        args.output = os.path.splitext(args.csv_file)[0] + '.mp4'
    
    # Create the animation
    create_animation(positions, thetas, pols_folder, args.output)

if __name__ == "__main__":
    main()