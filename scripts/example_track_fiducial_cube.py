import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from scripts_data_processing.data_processing.utils import aruco_utils
from scripts_data_processing.data_processing.utils.cv_utils import parse_pinhole_intrinsics
import yaml
import click

@click.command()
@click.option('-i', '--input_path', required=True, help='path to input video ending with .mp4')
@click.option('-o', '--output_path', required=True, help='output path ending with .mp4')
def main(input_path, output_path):
    # Record video with ArUco cube pose overlayed
    # load aruco config
    aruco_config = aruco_utils.parse_aruco_config(yaml.safe_load(open('config/aruco_config.yaml', 'r')))
    aruco_dict = aruco_config['aruco_dict']
    # Load camera intrinsics from calib.npz
    calib_file = 'config/gopro_intrinsics_pinhole.json'
    intrinsics_dict = parse_pinhole_intrinsics(yaml.safe_load(open(calib_file, 'r')))
    mtx = intrinsics_dict['K']
    dist = intrinsics_dict['D']
    # Cube params
    cube_params = aruco_utils.parse_aruco_cube_config(yaml.safe_load(open('config/cube_config.yaml', 'r')))
    aruco_utils.example_record_aruco_video(input_path, aruco_dict, mtx, dist, cube_params, output_path)

if __name__ == '__main__':
    main()