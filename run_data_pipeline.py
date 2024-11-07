"""
Main script for UMI data extraction pipeline.
python run_data_pipeline.py <session_dir>
"""

import sys
import os

ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess

# %%
@click.command()
@click.argument('session_dir', nargs=-1)
@click.option('-c', '--calibration_dir', type=str, default=None)
@click.option('-m', '--mode', type=str, required=True, help='pruner or gripper or pruner_inverse')
def main(session_dir, calibration_dir, mode):
    script_dir = pathlib.Path(__file__).parent.joinpath('scripts_data_processing')
    if calibration_dir is None:
        calibration_dir = pathlib.Path(__file__).parent.joinpath('config')
    else:
        calibration_dir = pathlib.Path(calibration_dir)
    assert calibration_dir.is_dir()

    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()

        print("############## 00_process_videos #############") # Both Cameras
        script_path = script_dir.joinpath("00_process_videos.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0
        
        print("############# 01_extract_gopro_imu ###########") # Gripper Camera
        script_path = script_dir.joinpath("01_extract_gopro_imu.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 01_extract_gopro_hilight ###########") # Gripper Camera
        script_path = script_dir.joinpath("01_extract_gopro_hilight.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input_dir', str(session.joinpath('demos')),
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 02_extract_aruco_cube_pose ###########") # External Camera
        script_path = script_dir.joinpath("02_extract_aruco_cube_pose.py")
        assert script_path.is_file()
        camera_intrinsics_pinhole = calibration_dir.joinpath('gopro_intrinsics_pinhole.json')
        aruco_config = calibration_dir.joinpath('aruco_config.yaml')
        cube_config = calibration_dir.joinpath('cube_config.yaml')
        assert camera_intrinsics_pinhole.is_file()
        assert aruco_config.is_file()
        assert cube_config.is_file()
        cmd = [
            'python', str(script_path),
            '--input_dir', str(session.joinpath('demos')),
            '--camera_intrinsics', str(camera_intrinsics_pinhole),
            '--aruco_yaml', str(aruco_config),
            '--cube_yaml', str(cube_config),
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 03_run_ekf ###########") # Both Cameras
        script_path = script_dir.joinpath("03_run_ekf.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input_dir', str(session.joinpath('demos')),
            '--mode', str(mode)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0
        
        print("############# 04_detect_aruco ###########") # Gripper Camera
        script_path = script_dir.joinpath("04_detect_aruco.py")
        assert script_path.is_file()
        camera_intrinsics_fisheye = calibration_dir.joinpath('gopro_intrinsics_2_7k.json')
        aruco_config = calibration_dir.joinpath('aruco_config.yaml')
        assert camera_intrinsics_fisheye.is_file()
        assert aruco_config.is_file()

        cmd = [
            'python', str(script_path),
            '--input_dir', str(session.joinpath('demos')),
            '--camera_intrinsics', str(camera_intrinsics_fisheye),
            '--aruco_yaml', str(aruco_config)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 05_run_calibrations ###########") # Gripper Camera
        script_path = script_dir.joinpath("05_run_calibrations.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            str(session),
            '--mode', str(mode) 
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 06_generate_dataset_plan ###########")
        script_path = script_dir.joinpath("06_generate_dataset_plan.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input', str(session),
            '--mode', str(mode) 
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0
  
## %%
if __name__ == "__main__":
    main()
