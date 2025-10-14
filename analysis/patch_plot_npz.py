#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import pandas as pd
import argparse
import glob
import os
from matplotlib.animation import FuncAnimation, FFMpegWriter

def load_npz_data(npz_file):
    """Load DDA simulation data from NPZ file and return structured data for a specific simulation."""
    data = np.load(npz_file, allow_pickle=True)
    
    # Print documentation if available
    if 'documentation' in data:
        print("NPZ File Documentation:")
        print(data['documentation'].item())
        print("\n" + "="*80 + "\n")
    
    # Get simulation information
    n_simulations = int(data['n_simulations'])
    folder_names = data['folder_names']
    
    print(f"Found {n_simulations} simulations in NPZ file:")
    for i, folder_name in enumerate(folder_names):
        print(f"  {i}: {folder_name}")
    
    return data, n_simulations, folder_names

def get_simulation_data(data, sim_index):
    """Extract data for a specific simulation from the NPZ file."""
    # Convert to zero-padded format for key lookup
    sim_key = f"sim_{sim_index:05d}"
    
    # Get basic simulation info
    frequencies = data[f'{sim_key}_frequencies']
    n_frequencies = len(frequencies)
    n_dipoles = int(data[f'{sim_key}_n_dipoles'])
    
    # Get dipole positions and orientations
    positions = np.column_stack([
        data[f'{sim_key}_x_positions'],
        data[f'{sim_key}_y_positions'], 
        data[f'{sim_key}_z_positions']
    ])
    thetas = data[f'{sim_key}_theta_orientations']
    
    # Get polarization data for both Ex and Mz components
    polarizations_ex = data[f'{sim_key}_polarizations_ex']
    polarizations_mz = data[f'{sim_key}_polarizations_mz']
    
    # Combine Ex and Mz into the same format as the original binary files
    # Original format: (N_frequencies, N_dipoles, 2) where 2 = [Ex, Mz]
    polarization_data = []
    for i, freq in enumerate(frequencies):
        # Get polarizations for this frequency
        ex_freq = polarizations_ex[i, :]  # shape: (n_dipoles,)
        mz_freq = polarizations_mz[i, :]  # shape: (n_dipoles,)
        
        # Stack to match original format: (n_dipoles, 2)
        combined_pols = np.column_stack([ex_freq, mz_freq])
        polarization_data.append((freq, combined_pols))
    
    return positions, thetas, polarization_data

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
    def __init__(self, positions, thetas, polarization_data, global_normalize=True, show_arrows=True, show_colorbar=True, colormap='viridis', plot_phase=False, plot_imaginary=False):
        self.positions = positions
        self.positions_microns = positions * 1e6  # store micron positions
        self.thetas = thetas
        self.polarization_data = polarization_data  # Already loaded from NPZ
        self.global_normalize = global_normalize
        self.show_arrows = show_arrows
        self.show_colorbar = show_colorbar
        self.plot_phase = plot_phase
        self.plot_imaginary = plot_imaginary
        self.colormap = plt.get_cmap(colormap)
        self.vor = Voronoi(positions[:, :2])
        self.regions, self.vertices = voronoi_finite_polygons_2d(self.vor)
        
        # Calculate arrow components if arrows are enabled
        if self.show_arrows:
            self.U = np.cos(thetas)
            self.V = np.sin(thetas)
        
        # Calculate global min/max for polarization data
        if self.plot_phase:
            # For phase, always use the fixed range -π to π
            self.global_min = -np.pi
            self.global_max = np.pi
        else:
            # For magnitude or imaginary, calculate actual min/max from data
            self.global_min = float('inf')
            self.global_max = float('-inf')
            
            for freq, polarizations in self.polarization_data:
                if self.plot_imaginary:
                    # Calculate min/max for imaginary part
                    px_values = np.imag(polarizations[:, 0])
                else:
                    # Calculate min/max for magnitude
                    px_values = np.abs(polarizations[:, 0])
                self.global_min = min(self.global_min, px_values.min())
                self.global_max = max(self.global_max, px_values.max())
        
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
            if self.plot_phase:
                label = 'Phase of $p_x$ (radians)'
            elif self.plot_imaginary:
                label = 'Im($p_x$) (C⋅m)'  # imaginary part of dipole moment
            else:
                label = '|$p_x$| (C⋅m)'  # magnitude of dipole moment
            self.cbar = self.fig.colorbar(self.smap, ax=self.ax, label=label)
        
        # Title for frequency display
        self.title = self.ax.set_title('')

        # Pre-create all polygon patches and store them
        self.patches = []
        for region in self.regions:
            polygon = self.vertices[region] * 1e6  # convert vertices to microns
            patch = plt.Polygon(polygon, alpha=1.0, edgecolor='black', linewidth=0.5)
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
        if self.plot_phase:
            colors = np.angle(polarizations[:, 0])
        elif self.plot_imaginary:
            colors = np.imag(polarizations[:, 0])
        else:
            colors = np.abs(polarizations[:, 0])
        normalized_colors = (colors - self.global_min) / (self.global_max - self.global_min)
        for patch, color in zip(self.patches, normalized_colors):
            patch.set_facecolor(self.colormap(color))
        
        freq_thz = freq / 1e12
        self.title_text = self.ax.set_title(f'Frequency: {freq_thz:.2f} THz')

    def init_animation(self):
        # Don't clear the axes, just return the artists we'll be animating
        return self.patches + [self.title_text]

    def animate(self, i):
        print(f'Animating frame {i+1}/{len(self.polarization_data)}')
        
        # Use pre-loaded polarization data
        freq, polarizations = self.polarization_data[i]
        
        # Calculate colors using magnitude, phase, or imaginary part
        if self.plot_phase:
            colors = np.angle(polarizations[:, 0])  # phase of px component
        elif self.plot_imaginary:
            colors = np.imag(polarizations[:, 0])   # imaginary part of px component
        else:
            colors = np.abs(polarizations[:, 0])    # magnitude of px component
        
        if self.global_normalize or self.plot_phase:
            # Always use global normalization for phase to maintain -π to π range
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

def create_animation(positions, thetas, polarization_data, output_file, global_normalize=True, 
                   show_arrows=True, show_colorbar=True, colormap='viridis', plot_phase=False, plot_imaginary=False):
    # Data is already sorted by frequency from get_simulation_data
    # Create animation with pre-loaded data
    anim = VoronoiAnimation(positions, thetas, polarization_data, 
                           global_normalize, show_arrows, show_colorbar, colormap, plot_phase, plot_imaginary)
    animation = FuncAnimation(anim.fig, anim.animate,
                            frames=len(polarization_data),
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
    parser = argparse.ArgumentParser(description='Create animated Voronoi diagram of polarization data from NPZ file')
    parser.add_argument('npz_file', help='NPZ file containing DDA simulation data')
    parser.add_argument('-s', '--simulation', type=int, default=0,
                       help='Index of simulation to plot (default: 0)')
    parser.add_argument('-o', '--output', 
                       help='Output video file (default: based on NPZ filename and simulation index)')
    parser.add_argument('--per-frame-normalize', action='store_true',
                       help='Normalize colors per frame instead of globally')
    parser.add_argument('--no-arrows', action='store_true',
                       help='Hide the orientation arrows in the plot')
    parser.add_argument('--no-colorbar', action='store_true',
                       help='Hide the colorbar in the plot')
    parser.add_argument('--plot-phase', action='store_true',
                       help='Plot phase of polarization instead of magnitude')
    parser.add_argument('--plot-imaginary', action='store_true',
                       help='Plot imaginary part of polarization instead of magnitude')
    parser.add_argument('--colormap', default=None,
                       help='Matplotlib colormap to use (default: viridis for magnitude, hsv for phase)')
    
    args = parser.parse_args()
    
    # Validate mutually exclusive options
    if args.plot_phase and args.plot_imaginary:
        raise ValueError("Cannot specify both --plot-phase and --plot-imaginary")
    
    # Load NPZ data
    if not os.path.exists(args.npz_file):
        raise FileNotFoundError(f"NPZ file not found: {args.npz_file}")
    
    data, n_simulations, folder_names = load_npz_data(args.npz_file)
    
    # Validate simulation index
    if args.simulation < 0 or args.simulation >= n_simulations:
        raise ValueError(f"Simulation index {args.simulation} out of range. Available indices: 0 to {n_simulations-1}")
    
    print(f"Processing simulation {args.simulation}: {folder_names[args.simulation]}")
    
    # Extract data for the specified simulation
    positions, thetas, polarization_data = get_simulation_data(data, args.simulation)
    
    print(f"Loaded simulation data:")
    print(f"  - {len(positions)} dipoles")
    print(f"  - {len(polarization_data)} frequency points")
    print(f"  - Frequency range: {polarization_data[0][0]/1e12:.2f} to {polarization_data[-1][0]/1e12:.2f} THz")
    
    # Set default colormap based on what is being plotted
    if args.colormap is None:
        if args.plot_phase:
            args.colormap = 'hsv'
        elif args.plot_imaginary:
            args.colormap = 'RdBu_r'  # Red-Blue colormap for imaginary values (can be positive or negative)
        else:
            args.colormap = 'viridis'
    
    # Set default output filename
    if args.output is None:
        base_name = os.path.splitext(args.npz_file)[0]
        sim_name = folder_names[args.simulation].replace('/', '_').replace('\\', '_')  # Safe filename
        flags = []
        if args.per_frame_normalize:
            flags.append('frame-norm')
        if args.no_arrows:
            flags.append('no-arrows')
        if args.no_colorbar:
            flags.append('no-cbar')
        if args.plot_phase:
            flags.append('phase')
        if args.plot_imaginary:
            flags.append('imaginary')
        # Check if non-default colormap is being used
        default_cmap = 'hsv' if args.plot_phase else ('RdBu_r' if args.plot_imaginary else 'viridis')
        if args.colormap != default_cmap:
            flags.append(f'cmap-{args.colormap}')
        
        flag_str = '_'.join(flags) if flags else 'default'
        args.output = f"{base_name}_sim{args.simulation:02d}_{sim_name}_{flag_str}.mp4"
    
    print(f"Creating animation: {args.output}")
    
    # Create the animation
    create_animation(positions, thetas, polarization_data, args.output, 
                    global_normalize=not args.per_frame_normalize,
                    show_arrows=not args.no_arrows,
                    show_colorbar=not args.no_colorbar,
                    colormap=args.colormap,
                    plot_phase=args.plot_phase,
                    plot_imaginary=args.plot_imaginary)
    
    print(f"Animation saved to: {args.output}")

if __name__ == "__main__":
    main()