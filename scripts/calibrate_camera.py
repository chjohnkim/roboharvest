import numpy as np
import cv2 as cv
import click
import json 

CHESSBOARD_SIZE = (9, 6) # This is the number of internal corners in the chessboard
SQUARE_SIZE = 0.025 # 25mm


@click.command()
@click.option('--video_path', required=True, help='Path to the video file containing the chessboard images')
def main(video_path):

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:,:2] = np.mgrid[:CHESSBOARD_SIZE[0],:CHESSBOARD_SIZE[1]].T.reshape(-1,2)
    objp *= SQUARE_SIZE

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.


    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    expected_frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    actual_frame_count = 0
    num_key_frames = 60
    actual_num_key_frames = 0
    while True:
        print(f"Processing frame {actual_frame_count}/{expected_frame_count}")
        ret, img = cap.read()
        if not ret:
            break
        else:
            actual_frame_count += 1
        if actual_frame_count % (expected_frame_count // num_key_frames) != 0:
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            actual_num_key_frames += 1
    
    print(f"Found {actual_num_key_frames}/{num_key_frames} key frames. Calibrating camera...")
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Compute reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print(f"Camera calibration complete.")
    print(f"Camera matrix:\n{mtx}")
    print(f"Distortion coefficients:\n{dist}")
    print( "Total error: {} pixels".format(mean_error/len(objpoints)) )

    intrinsics = {
        "final_reproj_error": mean_error/len(objpoints),
        "fps": cap.get(cv.CAP_PROP_FPS),
        "image_height": int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
        "image_width": int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
        "intrinsic_type": "PINHOLE",
        "intrinsics": {
            "focal_length_x": mtx[0, 0],
            "focal_length_y": mtx[1, 1],
            "principal_pt_x": mtx[0, 2],
            "principal_pt_y": mtx[1, 2],
            "skew": mtx[0, 1],
            "radial_distortion": dist[0].tolist(),
        },
        "nr_calib_images": actual_num_key_frames,
        "stabelized": False
    }
    cap.release()
    # Save the json file as json
    with open('config/gopro_intrinsics_pinhole.json', 'w') as f:
        json.dump(intrinsics, f, indent=4)
    # Save the camera calibration result for later use (we won't use rvecs / tvecs)
    #np.savez('config/camera_calib.npz', matrix=mtx, distortion=dist)
    
if __name__ == "__main__":
    main()

