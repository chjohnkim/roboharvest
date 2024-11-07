import json
import cv2
import av
from exiftool import ExifToolHelper
from utils.timecode_utils import mp4_get_start_datetime

MS_TO_S = 0.001

def load_telemetry(path_to_telemetry_file, vTimeStamps, coriTimeStamps, vAcc, vGyro):
    '''
    Output:
    vTimeStamps: list of timestamps in seconds for each accelerometer reading
    coriTimeStamps: list of timestamps in seconds for image frames(?)
    vAcc: list of 3D tuples representing accelerometer readings in m/s^2
    vGyro: list of 3D tuples representing gyroscope readings in rad/s

    # Usage example:
    vTimeStamps = []
    coriTimeStamps = []
    vAcc = []
    vGyro = []

    success = load_telemetry('imu_data.json', vTimeStamps, coriTimeStamps, vAcc, vGyro)

    if success:
        print("Telemetry loaded successfully")
    else:
        print("Failed to load telemetry")
    '''
    try:
        with open(path_to_telemetry_file, 'r') as file:
            j = json.load(file)
    except IOError:
        return False
    
    accl = j["1"]["streams"]["ACCL"]["samples"]
    gyro = j["1"]["streams"]["GYRO"]["samples"]
    cori = j["1"]["streams"]["CORI"]["samples"]
    
    sorted_acc = {}
    sorted_gyr = {}
    
    # Axis order: Z, X, Y as per: https://github.com/gopro/gpmf-parser
    for e in accl:
        v = (float(e["value"][1]), float(e["value"][2]), float(e["value"][0]))
        sorted_acc[float(e["cts"]) * MS_TO_S] = v        
    for e in gyro:
        v = (float(e["value"][1]), float(e["value"][2]), float(e["value"][0]))
        sorted_gyr[float(e["cts"]) * MS_TO_S] = v
    
    imu_start_t = min(sorted_acc.keys())
    for t, acc in sorted_acc.items():
        vTimeStamps.append(t - imu_start_t)
        vAcc.append(acc)
        
    for gyr in sorted_gyr.values():
        vGyro.append(gyr)
        
    for e in cori:
        coriTimeStamps.append(float(e["cts"]) * MS_TO_S)
    
    return True

def get_metadata(mp4_path):
    with ExifToolHelper() as et:
        meta = list(et.get_metadata(str(mp4_path)))[0]
        cam_serial = meta['QuickTime:CameraSerialNumber']
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
        meta_data = {
            'video_dir': mp4_path,
            'camera_serial': cam_serial,
            'start_date': start_date,
            'n_frames': n_frames,
            'fps': fps,
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp
        }
        return meta_data

# NOTE: This function is bad because it reads the entire video into memory. 
# Will stop using this function in the future.
def extract_images_with_timestamps(video_path, frames, frame_timestamps):
    '''
    Output:
    frames: list of frames extracted from the video
    frame_timestamps: list of timestamps in seconds for each frame

    # Usage example
    video_path = 'raw_video.mp4'
    frames = []
    frame_timestamps = []
    success = extract_images_with_timestamps(video_path, frames, frame_timestamps)
    if success:
        print(f"Extracted {len(frames)} frames with timestamps from {video_path}")
    else:
        print(f"Failed to extract frames with timestamps from {video_path}")
    '''
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get the current timestamp in milliseconds
        timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) * MS_TO_S
        frame_timestamps.append(timestamp_sec)
        # Store the frame
        frames.append(frame)
    cap.release()

    print(f"Expected {frame_count} frames, extracted {len(frames)} frames.")

    return True


if __name__ == '__main__':
    video_path = 'data/GX019393.MP4'
    frames = []
    frame_timestamps = []
    success = extract_images_with_timestamps(video_path, frames, frame_timestamps)
    if success:
        print(f"Extracted {len(frames)} frames with timestamps from {video_path}")
    else:
        print(f"Failed to extract frames with timestamps from {video_path}")