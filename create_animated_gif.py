#!/usr/bin/env python3
"""
Animated GIF Creator

This script takes a folder path and creates an animated GIF from all image files
in that folder, sorted by date modified. The GIF is saved within the same folder.

Usage:
    python create_animated_gif.py /path/to/folder
    
Or run interactively and it will prompt for folder path.

Requirements:
    pip install Pillow

Features:
- Supports common image formats: PNG, JPG, JPEG, GIF, BMP, TIFF
- Sorts images by date modified (oldest to newest)
- Customizable frame duration and loop settings
- Automatic output filename generation
- Progress indicator
- Error handling for corrupted images
"""

import os
import sys
import glob
from PIL import Image
import argparse
from datetime import datetime

def get_image_files(folder_path):
    """
    Get all image files in a folder, sorted by date modified
    
    Parameters:
    - folder_path: Path to the folder containing images
    
    Returns:
    - List of image file paths sorted by modification time
    """
    # Supported image extensions
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.tiff', '*.tif']
    
    image_files = []
    for ext in extensions:
        pattern = os.path.join(folder_path, ext)
        image_files.extend(glob.glob(pattern, recursive=False))
        # Also check uppercase
        pattern = os.path.join(folder_path, ext.upper())
        image_files.extend(glob.glob(pattern, recursive=False))
    
    # Remove duplicates (in case of case-insensitive filesystems)
    image_files = list(set(image_files))
    
    # Sort by modification time
    image_files.sort(key=lambda x: os.path.getmtime(x))
    
    return image_files

def resize_image_to_match(image, target_size):
    """
    Resize image to match target size while maintaining aspect ratio
    
    Parameters:
    - image: PIL Image object
    - target_size: Tuple of (width, height)
    
    Returns:
    - Resized PIL Image object
    """
    # Calculate scaling factor to fit within target size
    width_ratio = target_size[0] / image.width
    height_ratio = target_size[1] / image.height
    scale_factor = min(width_ratio, height_ratio)
    
    # Calculate new size
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    
    # Resize image
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new image with target size and paste resized image in center
    result = Image.new('RGB', target_size, (255, 255, 255))  # White background
    
    # Calculate position to center the image
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2
    
    result.paste(resized, (x_offset, y_offset))
    
    return result

def create_animated_gif(folder_path, output_filename=None, duration=500, loop=0, max_size=(800, 600)):
    """
    Create an animated GIF from images in a folder
    
    Parameters:
    - folder_path: Path to folder containing images
    - output_filename: Name of output GIF file (default: auto-generated)
    - duration: Duration of each frame in milliseconds
    - loop: Number of loops (0 = infinite)
    - max_size: Maximum size for frames (width, height)
    
    Returns:
    - Path to created GIF file
    """
    # Get image files
    image_files = get_image_files(folder_path)
    
    if not image_files:
        raise ValueError(f"No image files found in {folder_path}")
    
    print(f"Found {len(image_files)} image files:")
    for i, img_file in enumerate(image_files, 1):
        mod_time = datetime.fromtimestamp(os.path.getmtime(img_file))
        print(f"  {i:2d}. {os.path.basename(img_file)} ({mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
    
    # Generate output filename if not provided
    if output_filename is None:
        folder_name = os.path.basename(os.path.abspath(folder_path))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{folder_name}_animation_{timestamp}.gif"
    
    output_path = os.path.join(folder_path, output_filename)
    
    # Load and process images
    print(f"\nProcessing images...")
    processed_images = []
    
    for i, img_file in enumerate(image_files, 1):
        try:
            print(f"  Processing {i}/{len(image_files)}: {os.path.basename(img_file)}")
            
            # Load image
            with Image.open(img_file) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to fit within max_size
                processed_img = resize_image_to_match(img, max_size)
                processed_images.append(processed_img)
                
        except Exception as e:
            print(f"    Warning: Could not process {img_file}: {e}")
            continue
    
    if not processed_images:
        raise ValueError("No images could be processed successfully")
    
    # Create animated GIF
    print(f"\nCreating animated GIF...")
    print(f"  Output: {output_path}")
    print(f"  Frame duration: {duration}ms")
    print(f"  Loop count: {'infinite' if loop == 0 else loop}")
    print(f"  Frame size: {max_size[0]}x{max_size[1]}")
    
    # Save as animated GIF
    processed_images[0].save(
        output_path,
        save_all=True,
        append_images=processed_images[1:],
        duration=duration,
        loop=loop,
        optimize=True
    )
    
    # Get file size
    file_size = os.path.getsize(output_path)
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"\nAnimated GIF created successfully!")
    print(f"  File: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Frames: {len(processed_images)}")
    
    return output_path

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Create animated GIF from images in a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_animated_gif.py /path/to/images
  python create_animated_gif.py /path/to/images --duration 1000 --max-size 1024 768
  python create_animated_gif.py /path/to/images --output my_animation.gif --no-loop
        """
    )
    
    parser.add_argument('folder', nargs='?', help='Path to folder containing images')
    parser.add_argument('--duration', '-d', type=int, default=500,
                       help='Duration of each frame in milliseconds (default: 500)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output filename (default: auto-generated)')
    parser.add_argument('--loop', type=int, default=0,
                       help='Number of loops (0 = infinite, default: 0)')
    parser.add_argument('--no-loop', action='store_true',
                       help='Disable looping (same as --loop 1)')
    parser.add_argument('--max-size', nargs=2, type=int, default=[800, 600],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Maximum frame size in pixels (default: 800 600)')
    
    args = parser.parse_args()
    
    # Get folder path
    if args.folder:
        folder_path = args.folder
    else:
        # Interactive mode
        folder_path = input("Enter folder path containing images: ").strip()
    
    # Validate folder path
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory")
        sys.exit(1)
    
    # Handle loop setting
    loop_count = args.loop
    if args.no_loop:
        loop_count = 1
    
    try:
        # Create animated GIF
        output_path = create_animated_gif(
            folder_path=folder_path,
            output_filename=args.output,
            duration=args.duration,
            loop=loop_count,
            max_size=tuple(args.max_size)
        )
        
        print(f"\n{'='*60}")
        print("SUCCESS: Animated GIF created!")
        print(f"Location: {output_path}")
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
