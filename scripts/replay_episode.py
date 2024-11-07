import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import zarr
import click
from diffusion_policy.common.replay_buffer import ReplayBuffer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

@click.command()
@click.option('--zarr_path', '-z', required=True, help='Path to zarr file')
def main(zarr_path):
    group = zarr.open(zarr_path, mode='a')
    replay_buffer = ReplayBuffer.create_from_group(group)
    print(replay_buffer)
    action = np.array(replay_buffer.data['action'][:])
    pos = np.array(replay_buffer.data['robot0_eef_pos'][:])
    episode_ends = np.array(replay_buffer.meta['episode_ends'][:])

    # Plot action in red and pos in blue, but make the colors get stronger as the episode progresses
    for i in range(len(episode_ends)):
        start = 0 if i == 0 else episode_ends[i-1]
        end = episode_ends[i]
        print(f"Episode {i}: {start} to {end}")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Use colormap to make colors get stronger as episode progresses
        colors = np.linspace(0, 1, end-start)
        action_cmap = mpl.colormaps.get_cmap('Reds')
        pos_cmap = mpl.colormaps.get_cmap('Blues')
        ax.scatter(action[start:end, 0], action[start:end, 1], action[start:end, 2], c=action_cmap(colors))
        ax.scatter(pos[start:end, 0], pos[start:end, 1], pos[start:end, 2], c=pos_cmap(colors))

        # Add coordinate frame at origin
        ax.quiver(0, 0, 0, 0.1, 0, 0, color='r')
        ax.quiver(0, 0, 0, 0, 0.1, 0, color='g')
        ax.quiver(0, 0, 0, 0, 0, 0.1, color='b')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Episode {i}")

        # Set equal scaling
        x_limits = [np.min([np.min(action[:, 0]), np.min(pos[:, 0])]), np.max([np.max(action[:, 0]), np.max(pos[:, 0])])]
        y_limits = [np.min([np.min(action[:, 1]), np.min(pos[:, 1])]), np.max([np.max(action[:, 1]), np.max(pos[:, 1])])]
        z_limits = [np.min([np.min(action[:, 2]), np.min(pos[:, 2])]), np.max([np.max(action[:, 2]), np.max(pos[:, 2])])]
        max_range = np.array([x_limits[1]-x_limits[0], y_limits[1]-y_limits[0], z_limits[1]-z_limits[0]]).max() / 2.0

        mid_x = (x_limits[1] + x_limits[0]) * 0.5
        mid_y = (y_limits[1] + y_limits[0]) * 0.5
        mid_z = (z_limits[1] + z_limits[0]) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Add legend
        red_patch = mpl.patches.Patch(color='red', label='Commanded Action')
        blue_patch = mpl.patches.Patch(color='blue', label='Actual Position')
        plt.legend(handles=[red_patch, blue_patch])
        plt.show()

if __name__ == '__main__':
    main()
