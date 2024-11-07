import os
import pathlib
import click
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

def calculate_grid_size(num_images):
    """Calculate the best rows and columns to fit num_images in a grid."""
    # Start from the square root to find the closest grid size
    for i in range(int(math.sqrt(num_images)), 0, -1):
        cols = math.ceil(num_images / i)  # Columns is the ceiling of images divided by rows
        if i * cols >= num_images:
            return i, cols
    return num_images, 1  # Fall back to a single row if no better option

def visualize_images_in_grid(png_files, input_dir):
    """Visualize the PNG files in a grid using matplotlib."""
    if not png_files:
        print("No PNG files to display.")
        return

    num_images = len(png_files)
    rows, cols = calculate_grid_size(num_images)

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    # Flatten the axes array if necessary
    axes = axes.flatten() if num_images > 1 else [axes]

    for ax, img_path in zip(axes, png_files):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('off')  # Hide axes for a cleaner view

    # Hide any remaining empty subplots (if there are fewer images than grid cells)
    for ax in axes[num_images:]:
        ax.axis('off')

    # Add title with the directory name
    fig.suptitle(input_dir.name, fontsize=16)
    
    # Display the grid of images
    plt.tight_layout()
    plt.show()

# %%
@click.command()
@click.argument('session_dir', nargs=-1)
def main(session_dir):
    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()
        demos_dir = session.joinpath('demos')
        input_dirs = [x.parent for x in demos_dir.glob('*/_ekf_all.png')]
        print(f'Found {len(input_dirs)} video dirs')
        for input_dir in input_dirs:
            # Get all png files and visualize them in a grid
            print(f'Processing {input_dir}')
            png_files = list(input_dir.glob('*.png'))
            
            # Sort the PNG files by name
            png_files.sort(key=lambda x: x.name)

            # Visualize the images in a grid
            visualize_images_in_grid(png_files, input_dir)


if __name__ == '__main__':
    main()