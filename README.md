# RoboHarvest
This repository is the codebase for our paper: [Title of Paper]().

See the [project page]() for more information.

## 🙏 Acknowledgement
This repository is adapted from the [Universal Manipulation Interface (UMI)](https://github.com/real-stanford/universal_manipulation_interface) repository with modifications in its data collection method, processing pipeline, and hardware customization.   

## 🛠️ Installation 
Only tested on Ubuntu 22.04

Install docker following the [official documentation](https://docs.docker.com/engine/install/ubuntu/) and finish [linux-postinstall](https://docs.docker.com/engine/install/linux-postinstall/).

Install system-level dependencies:
```
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

Create conda environment:
```
conda env create -f conda_environment.yaml
```

Activate environment:
```
conda activate roboharvest 
```

Install DynamixelSDK for gripper control by following the [official documentation](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/download/#repository).

## 🏃 Running Data Preprocessing Pipeline
Download example data and unzip all video's into a single folder (e.g. `roboharvest/demo_data`):
```
pip install gdown
gdown 'https://drive.google.com/uc?export=download&id=1AEcEIvVZTcqkem8oE1XAGe1M05n1VJ_t'
unzip demo_data.zip -d demo_data && rm demo_data.zip
```

Run data pipeline:
```
python run_data_pipeline.py ./demo_data --mode pruner_inverse
```

Generate dataset for training.
```
python scripts_data_processing/07_generate_replay_buffer.py -o ./demo_data/dataset.zarr.zip ./demo_data
```

## 🚆 Training Diffusion Policy
Single-GPU training. 
```
python train.py --config-name=train_diffusion_unet_timm_umi_workspace task.dataset_path=demo_data/dataset.zarr.zip
```

Multi-GPU training.
```
accelerate --num_processes <ngpus> train.py --config-name=train_diffusion_unet_timm_umi_workspace task.dataset_path=demo_data/dataset.zarr.zip
```

## 💡 Other useful scripts
Once data is processed, 6-DOF pose can be visualized by specifying data directory:
```
python scripts/visualize_data.py ./demo_data
```

To visualize a rollout trajectory (only works on output zarr files from actual robot deployment rollout):
```
python scripts/replay_episode.py -z [./data/path_to_zarr]
```

To generate fiducial cube tracking video:
```
python scripts/example_track_fiducial_cube.py -i [path_to_mp4_file] -o [path_to_output_mp4]
```

## 🦾 Real World Deployment
We refer users to [UMI](https://github.com/real-stanford/universal_manipulation_interface) for hardware setup of the UR5e Robot Arm, GoPro camera, and SpaceMouse. The only difference is the custom shear-gripper end-effector detailed in the hardware section. 
Make appropriate changes to `robot_launch.sh` according to directories and USB settings and launch:
```
bash robot_launch.sh
```

## Data Collection
### Calibrate bird-eye view pinhole (linear) camera
Scan [this GoPro firmware QR code](assets/QR_gopro_linear_mVr1440p60e0!NfL0hS0dR0aSv1q0oV0R1L0.png), take video of checkerboard, and run:
```
python scripts/calibrate_camera.py --video_path [path_to_video.mp4]
```

# Coordinate Frames
<p align="center">
  <img src="assets/gopro_coordinate_frame.png" alt="GOPRO Coordintate Frame" height="350"/>
  <img src="assets/cube_coordinate_frame.png" alt="Cube Coordintate Frame" height="350" />
</p>

# Useful Links
- [OpenImuCameraCalibrator](https://github.com/urbste/OpenImuCameraCalibrator): GoPro Calibration (frames and syncs)
- [gpmf_parser](https://gopro.github.io/gpmf-parser/): IMU Frame

## 🏷️ License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.