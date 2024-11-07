import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
from tqdm import tqdm
import yaml
import av
import numpy as np
import cv2
import pickle

from utils.cv_utils import parse_pinhole_intrinsics
from utils.aruco_utils import (
    parse_aruco_config, 
    parse_aruco_cube_config,
    extract_aruco_cube_pose_from_video
)

# %%
@click.command()
@click.option('-i', '--input', required=True)
@click.option('-o', '--output', required=True)
@click.option('-in', '--intrinsics_json', required=True)
@click.option('-ay', '--aruco_yaml', required=True)
@click.option('-ay', '--cube_yaml', required=True)
@click.option('-n', '--num_workers', type=int, default=4)
def main(input, output, intrinsics_json, aruco_yaml, cube_yaml, num_workers):
    cv2.setNumThreads(num_workers)
    # load aruco config
    aruco_config = parse_aruco_config(yaml.safe_load(open(aruco_yaml, 'r')))
    aruco_dict = aruco_config['aruco_dict']
    #marker_size_map = aruco_config['marker_size_map']
    
    # load cube params
    cube_params = parse_aruco_cube_config(yaml.safe_load(open(cube_yaml, 'r')))

    # load intrinsics
    intrinsics_dict = parse_pinhole_intrinsics(yaml.safe_load(open(intrinsics_json, 'r')))
    mtx, dist = intrinsics_dict['K'], intrinsics_dict['D']

    cube_poses = extract_aruco_cube_pose_from_video(input, aruco_dict, mtx, dist, cube_params)

    # dump
    pickle.dump(cube_poses, open(os.path.expanduser(output), 'wb'))

# %%
if __name__ == "__main__":
    main()
