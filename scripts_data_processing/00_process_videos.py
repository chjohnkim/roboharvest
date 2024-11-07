import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import shutil
from exiftool import ExifToolHelper
from data_processing.utils.timecode_utils import mp4_get_start_datetime
import cv2
import av 
import pandas as pd 

# %%
@click.command(help='Session directories. Assumming mp4 videos are in <session_dir>/raw_videos')
@click.argument('session_dir', nargs=-1)
def main(session_dir):
    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()
        # hardcode subdirs
        input_dir = session.joinpath('raw_videos')
        output_dir = session.joinpath('demos')
        
        # create raw_videos if don't exist
        if not input_dir.is_dir():
            input_dir.mkdir()
            print(f"{input_dir.name} subdir don't exits! Creating one and moving all mp4 videos inside.")
            for mp4_path in list(session.glob('**/*.MP4')) + list(session.glob('**/*.mp4')):
                out_path = input_dir.joinpath(mp4_path.name)
                shutil.move(mp4_path, out_path)


        # look for mp4 video in all subdirectories in input_dir
        input_mp4_paths = list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4'))
        print(f'Found {len(input_mp4_paths)} MP4 videos')

        # Get video metadata
        cam_serial_to_meta = {}
        with ExifToolHelper() as et:
            for mp4_path in input_mp4_paths:
                if mp4_path.is_symlink():
                    print(f"Skipping {mp4_path.name}, already moved.")
                    continue
                if mp4_path.name.startswith('gripper_cal') or mp4_path.parent.name.startswith('gripper_cal'):
                    print(f"Skipping {mp4_path.name}, gripper_cal video.")
                    continue

                meta = list(et.get_metadata(str(mp4_path)))[0]
                cam_serial = meta['QuickTime:CameraSerialNumber']
                if cam_serial not in cam_serial_to_meta:
                    cam_serial_to_meta[cam_serial] = []
                start_date = mp4_get_start_datetime(str(mp4_path))
                start_timestamp = start_date.timestamp()
                
                cap = cv2.VideoCapture(str(mp4_path))
                fps_cv2 = cap.get(cv2.CAP_PROP_FPS)
                n_frames_cv2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                with av.open(str(mp4_path), 'r') as container:
                    stream = container.streams.video[0]
                    n_frames = stream.frames
                    fps = stream.average_rate
                    if (float(fps)!= float(fps_cv2)) or (n_frames!=n_frames_cv2):
                        print(f"Inconsistent video reading: {float(fps)}|{n_frames} vs {fps_cv2}|{n_frames_cv2} in {mp4_path}")
                        exit(1)
                duration_sec = float(n_frames / fps)
                end_timestamp = start_timestamp + duration_sec
                
                cam_serial_to_meta[cam_serial].append({
                    'video_dir': mp4_path,
                    'camera_serial': cam_serial,
                    'start_date': start_date,
                    'n_frames': n_frames,
                    'fps': fps,
                    'start_timestamp': start_timestamp,
                    'end_timestamp': end_timestamp
                })

        # Convert to DataFrame
        for key, value in cam_serial_to_meta.items():
            cam_serial_to_meta[key] = pd.DataFrame(data=value)
        
        print(f'Found {len(cam_serial_to_meta)} cameras')
        if len(cam_serial_to_meta) == 0:
            print('No videos found!')
            continue
        
        # Get the first key
        cam_serial =  list(cam_serial_to_meta.keys())[0]
        events = []
        for d1 in cam_serial_to_meta[cam_serial].itertuples(index=True):
            event = {cam_serial: d1.Index}
            start_timestamp = d1.start_timestamp
            end_timestamp = d1.end_timestamp
            for key in cam_serial_to_meta.keys():
                if key == cam_serial:
                    continue
                for d2 in cam_serial_to_meta[key].itertuples(index=True):
                    max_start = max(start_timestamp, d2.start_timestamp)
                    min_end = min(end_timestamp, d2.end_timestamp)
                    if max_start < min_end:
                        assert key not in event
                        # Store index of the event
                        event[key] = d2.Index
                        
            events.append(event)
        
        # To manually check filter out events with missing cameras
        # However, this should be automatically handled by the code
        #for event in events:
        #    if len(event) != len(cam_serial_to_meta):
        #        for key, value in event.items():
        #            print(cam_serial_to_meta[key].loc[value].video_dir)
        #import sys; sys.exit()

        # Filter out events with missing cameras
        events = [event for event in events if len(event) == len(cam_serial_to_meta)]
        
        # Compute the sum of the duration of each camera in events
        cam_serial_to_duration = {}
        for event in events:
            for key, value in event.items():
                if key not in cam_serial_to_duration:
                    cam_serial_to_duration[key] = 0
                data = cam_serial_to_meta[key].loc[value]
                duration = data.end_timestamp - data.start_timestamp
                cam_serial_to_duration[key] += duration

        # Determine the birds eye camera as it has 1 less event than the others
        cam_serial_event_count = {key: len(value) for key, value in cam_serial_to_meta.items()}
        birds_eye_cam_serial = min(cam_serial_event_count, key=cam_serial_event_count.get)

        for event in events:
            birds_eye_video = cam_serial_to_meta[birds_eye_cam_serial].loc[event[birds_eye_cam_serial]].video_dir
            for key, value in event.items():
                if key == birds_eye_cam_serial:
                    continue
                # Move Gripper Video
                gripper_video = cam_serial_to_meta[key].loc[value].video_dir

                start_date = mp4_get_start_datetime(str(gripper_video))
                meta = list(et.get_metadata(str(gripper_video)))[0]
                cam_serial = meta['QuickTime:CameraSerialNumber']
                out_dname = 'demo_' + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")

                # create directory
                this_out_dir = output_dir.joinpath(out_dname)
                this_out_dir.mkdir(parents=True, exist_ok=True)
                
                # move videos
                vfname = 'raw_video.mp4'
                out_video_path = this_out_dir.joinpath(vfname)
                shutil.move(gripper_video, out_video_path)

                # create symlink back from original location
                # relative_to's walk_up argument is not avaliable until python 3.12
                dots = os.path.join(*['..'] * len(gripper_video.parent.relative_to(session).parts))
                rel_path = str(out_video_path.relative_to(session))
                symlink_path = os.path.join(dots, rel_path)                
                gripper_video.symlink_to(symlink_path)

                # Move Birdseye Video
                vfname = 'birdseye_video.mp4'
                out_video_path = this_out_dir.joinpath(vfname)
                shutil.move(birds_eye_video, out_video_path)

                # create symlink back from original location
                # relative_to's walk_up argument is not avaliable until python 3.12
                dots = os.path.join(*['..'] * len(birds_eye_video.parent.relative_to(session).parts))
                rel_path = str(out_video_path.relative_to(session))
                symlink_path = os.path.join(dots, rel_path)                
                birds_eye_video.symlink_to(symlink_path)
        
        # create gripper calibration video if don't exist
        gripper_cam_serial = [key for key in cam_serial_to_duration if key!=birds_eye_cam_serial]
        gripper_cal_dir = input_dir.joinpath('gripper_calibration')
        if not gripper_cal_dir.is_dir():
            gripper_cal_dir.mkdir()
            print("raw_videos/gripper_calibration don't exist! Creating one with the first video of each camera serial.")
            
            serial_start_dict = dict()
            serial_path_dict = dict()
            with ExifToolHelper() as et:
                for mp4_path in list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4')):
                    
                    start_date = mp4_get_start_datetime(str(mp4_path))
                    meta = list(et.get_metadata(str(mp4_path)))[0]
                    cam_serial = meta['QuickTime:CameraSerialNumber']
                    
                    if cam_serial in serial_start_dict:
                        if start_date < serial_start_dict[cam_serial]:
                            serial_start_dict[cam_serial] = start_date
                            serial_path_dict[cam_serial] = mp4_path
                    else:
                        serial_start_dict[cam_serial] = start_date
                        serial_path_dict[cam_serial] = mp4_path
            
            for serial, path in serial_path_dict.items():
                if serial in gripper_cam_serial:
                    print(f"Selected {path.name} for camera serial {serial}")
                    out_path = gripper_cal_dir.joinpath(path.name)
                    shutil.move(path, out_path)

        # Move calibration videos
        input_mp4_paths = list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4'))
        unused_videos = []
        with ExifToolHelper() as et:
            for mp4_path in input_mp4_paths:
                if mp4_path.is_symlink():
                    continue
                start_date = mp4_get_start_datetime(str(mp4_path))
                meta = list(et.get_metadata(str(mp4_path)))[0]
                cam_serial = meta['QuickTime:CameraSerialNumber']
                out_dname = 'demo_' + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")

                if mp4_path.name.startswith('gripper_cal') or mp4_path.parent.name.startswith('gripper_cal'):
                    out_dname = "gripper_calibration_" + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")
                else:
                    unused_videos.append(mp4_path)
                    continue 
                # create directory
                this_out_dir = output_dir.joinpath(out_dname)
                this_out_dir.mkdir(parents=True, exist_ok=True)
                
                # move videos
                vfname = 'raw_video.mp4'
                out_video_path = this_out_dir.joinpath(vfname)
                shutil.move(mp4_path, out_video_path)

                # create symlink back from original location
                # relative_to's walk_up argument is not avaliable until python 3.12
                dots = os.path.join(*['..'] * len(mp4_path.parent.relative_to(session).parts))
                rel_path = str(out_video_path.relative_to(session))
                symlink_path = os.path.join(dots, rel_path)                
                mp4_path.symlink_to(symlink_path)
        print(f'Found {len(unused_videos)} unused videos: {unused_videos}')        
# %%
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main.main(['--help'])
    else:
        main()
