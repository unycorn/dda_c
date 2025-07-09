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
        radius = np.ptp(vor.points, axis=0).max() * 2  # Updated to use np.ptp

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
    def __init__(self, positions, thetas, pols_files, global_normalize=True, show_arrows=True, show_colorbar=True, colormap='viridis'):
        self.positions = positions
        self.positions_microns = positions * 1e6  # store micron positions
        self.thetas = thetas
        self.pols_files = pols_files
        self.global_normalize = global_normalize
        self.show_arrows = show_arrows
        self.show_colorbar = show_colorbar
        self.colormap = plt.get_cmap(colormap)
        self.vor = Voronoi(positions[:, :2])
        self.regions, self.vertices = voronoi_finite_polygons_2d(self.vor)
        
        # Calculate arrow components if arrows are enabled
        if self.show_arrows:
            self.U = np.cos(thetas)
            self.V = np.sin(thetas)
        
        # Load all polarization data up front
        self.polarization_data = []
        self.global_min = float('inf')
        self.global_max = float('-inf')
        for pols_file in pols_files:
            N, freq, polarizations = read_polarizations_binary(pols_file)
            self.polarization_data.append((freq, polarizations))
            px_mag = np.abs(polarizations[:, 0])
            self.global_min = min(self.global_min, px_mag.min())
            self.global_max = max(self.global_max, px_mag.max())
        
        # Setup the figure
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # Set plot limits with tighter margins
        margin = 0.01
        x_min, x_max = self.positions_microns[:, 0].min(), self.positions_microns[:, 0].max()
        y_min, y_max = self.positions_microns[:, 1].min(), self.positions_microns[:, 1].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        self.ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
        self.ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
        
        # Set plot properties
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (µm)')
        self.ax.set_ylabel('Y (µm)')
        
        # Initialize colorbar only if enabled
        self.smap = plt.cm.ScalarMappable(cmap=self.colormap,
                                         norm=plt.Normalize(self.global_min, self.global_max))
        if self.show_colorbar:
            self.cbar = self.fig.colorbar(self.smap, ax=self.ax, 
                                        label='|$p_x$| (C⋅m)')  # dipole moment units
        
        # Title for frequency display
        self.title = self.ax.set_title('')

        # Pre-create all polygon patches and store them
        self.patches = []
        for region in self.regions:
            polygon = self.vertices[region] * 1e6  # convert vertices to microns
            patch = plt.Polygon(polygon, alpha=0.6, edgecolor='black', linewidth=0.5)
            self.ax.add_patch(patch)
            self.patches.append(patch)
        
        # Pre-create quiver if arrows are enabled
        if self.show_arrows:
            self.quiver = self.ax.quiver(self.positions_microns[:, 0], 
                                       self.positions_microns[:, 1],
                                       self.U, self.V, color='black',
                                       scale=100, width=0.001,
                                       headwidth=3, headlength=4,
                                       headaxislength=3.5,
                                       pivot='middle')
        
        # Set initial frame
        freq, polarizations = self.polarization_data[0]
        colors = np.abs(polarizations[:, 0])
        normalized_colors = (colors - self.global_min) / (self.global_max - self.global_min)
        for patch, color in zip(self.patches, normalized_colors):
            patch.set_facecolor(plt.cm.viridis(color))
        
        freq_thz = freq / 1e12
        self.title_text = self.ax.set_title(f'Frequency: {freq_thz:.2f} THz')

    def init_animation(self):
        # Don't clear the axes, just return the artists we'll be animating
        return self.patches + [self.title_text]

    def animate(self, i):
        print(f'Animating frame {i+1}/{len(self.polarization_data)}')
        
        # Use pre-loaded polarization data
        freq, polarizations = self.polarization_data[i]
        
        # Calculate colors using actual magnitudes
        colors = np.abs(polarizations[:, 0])  # px component
        
        if self.global_normalize:
            normalized_colors = (colors - self.global_min) / (self.global_max - self.global_min)
            vmin, vmax = self.global_min, self.global_max
        else:
            vmin, vmax = colors.min(), colors.max()
            normalized_colors = (colors - vmin) / (vmax - vmin)
        
        # Update colorbar array and limits
        self.smap.set_array(colors)
        self.smap.set_clim(vmin, vmax)
        
        # Update polygon colors without recreating them
        for patch, color in zip(self.patches, normalized_colors):
            patch.set_facecolor(self.colormap(color))
        
        # Update frequency title
        freq_thz = freq / 1e12
        self.title_text.set_text(f'Frequency: {freq_thz:.2f} THz')
        
        # Maintain tight plot limits
        margin = 0.01
        x_min, x_max = self.positions_microns[:, 0].min(), self.positions_microns[:, 0].max()
        y_min, y_max = self.positions_microns[:, 1].min(), self.positions_microns[:, 1].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        self.ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
        self.ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
        
        # Return all artists that were modified
        return self.patches + [self.title_text]

def create_animation(positions, thetas, pols_folder, output_file, global_normalize=True, 
                   show_arrows=True, show_colorbar=True, colormap='viridis'):
    # Get all .pols files
    pols_files = glob.glob(os.path.join(pols_folder, '*.pols'))
    
    # Read frequencies and full data to sort files
    data_pairs = []
    for file in pols_files:
        N, freq, polarizations = read_polarizations_binary(file)
        data_pairs.append((freq, file, N, polarizations))
    
    # Sort by frequency
    data_pairs.sort()
    sorted_pols_files = [pair[1] for pair in data_pairs]
    
    # Create animation with pre-loaded data
    anim = VoronoiAnimation(positions, thetas, sorted_pols_files, 
                           global_normalize, show_arrows, show_colorbar, colormap)
    animation = FuncAnimation(anim.fig, anim.animate,
                            frames=len(sorted_pols_files),
                            init_func=anim.init_animation,
                            interval=300,
                            blit=True)
    
    # Save animation with more stable settings
    writer = FFMpegWriter(fps=20,
                         metadata=dict(artist='Me'),
                         bitrate=2000,  # Reduced bitrate for stability
                         codec='libx264',  # Explicitly use libx264 codec
                         extra_args=['-preset', 'medium',  # Use medium preset instead of veryslow
                                   '-crf', '23',  # Slightly reduced quality but more stable
                                   '-pix_fmt', 'yuv420p',  # Ensure compatibility
                                   '-movflags', '+faststart',  # Enable streaming-friendly format
                                   '-profile:v', 'main',  # Use main profile for better compatibility
                                   '-tune', 'animation'])  # Optimize for animated content
    animation.save(output_file, writer=writer)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Create animated Voronoi diagram of polarization data')
    parser.add_argument('csv_file', help='CSV file with x,y,z,theta columns')
    parser.add_argument('-o', '--output', 
                       help='Output video file (default: same name as input CSV but with .mp4 extension)')
    parser.add_argument('--per-frame-normalize', action='store_true',
                       help='Normalize colors per frame instead of globally')
    parser.add_argument('--no-arrows', action='store_true',
                       help='Hide the orientation arrows in the plot')
    parser.add_argument('--no-colorbar', action='store_true',
                       help='Hide the colorbar in the plot')
    parser.add_argument('--colormap', default='viridis',
                       help='Matplotlib colormap to use (default: viridis)')
    
    args = parser.parse_args()
    
    # Read position data
    df = pd.read_csv(args.csv_file)
    positions = df[['x', 'y', 'z']].values
    thetas = df['theta'].values
    
    # Get the pols folder path from the CSV path
    csv_dir = os.path.dirname(args.csv_file)
    csv_basename = os.path.basename(args.csv_file)
    # Check if this is a cdm_input file in an l{l_val} directory
    if csv_basename.startswith('cdm_input_') and os.path.basename(csv_dir).startswith('l'):
        pols_folder = csv_dir
    else:
        # Fall back to old behavior of removing .csv extension
        pols_folder = os.path.splitext(args.csv_file)[0]
    
    if not os.path.isdir(pols_folder):
        raise ValueError(f"Could not find polarization data folder: {pols_folder}")
    
    # Set default output filename to be the same as input but with flags and .mp4 extension
    if args.output is None:
        base_name = os.path.splitext(args.csv_file)[0]
        flags = []
        if args.per_frame_normalize:
            flags.append('frame-norm')
        if args.no_arrows:
            flags.append('no-arrows')
        if args.no_colorbar:
            flags.append('no-cbar')
        if args.colormap != 'viridis':
            flags.append(f'cmap-{args.colormap}')
        
        flag_str = '_'.join(flags) if flags else 'default'
        args.output = f"{base_name}_{flag_str}.mp4"
    
    # Create the animation
    create_animation(positions, thetas, pols_folder, args.output, 
                    global_normalize=not args.per_frame_normalize,
                    show_arrows=not args.no_arrows,
                    show_colorbar=not args.no_colorbar,
                    colormap=args.colormap)

if __name__ == "__main__":
    main()