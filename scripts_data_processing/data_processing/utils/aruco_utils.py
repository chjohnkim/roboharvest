import cv2
import numpy as np
import time
from skimage.metrics import structural_similarity as ssim

S_TO_MS = 1000.0

################# Parser Functions #################
def parse_aruco_config(aruco_config_dict: dict):
    """
    example:
    aruco_dict:
        predefined: DICT_4X4_50
    marker_size_map: # all unit in meters
        default: 0.15
        12: 0.2
    """
    aruco_dict = get_aruco_dict(**aruco_config_dict['aruco_dict'])

    n_markers = len(aruco_dict.bytesList)
    marker_size_map = aruco_config_dict['marker_size_map']
    default_size = marker_size_map.get('default', None)
    
    out_marker_size_map = dict()
    for marker_id in range(n_markers):
        size = default_size
        if marker_id in marker_size_map:
            size = marker_size_map[marker_id]
        out_marker_size_map[marker_id] = size
    
    result = {
        'aruco_dict': aruco_dict,
        'marker_size_map': out_marker_size_map
    }
    return result

def parse_aruco_cube_config(aruco_cube_config_dict: dict):
    inner = aruco_cube_config_dict['marker_sep']/2.0
    outer = inner + aruco_cube_config_dict['marker_dim']
    half_cube = aruco_cube_config_dict['cube_dim']/2.0
    side_to_ids = aruco_cube_config_dict['cube_side2id_map']
    # Local coordinate system of face. This depends on how you define the coordinate system of the cube
    face_coords = np.array([[
        [-outer, outer, 0],
        [-inner, outer, 0],
        [-inner, inner, 0],
        [-outer, inner, 0]],
        [[inner, outer, 0],
        [outer, outer, 0],
        [outer, inner, 0],
        [inner, inner, 0]],
        [[-outer, -inner, 0],
        [-inner, -inner, 0],
        [-inner, -outer, 0],
        [-outer, -outer, 0]],
        [[inner, -inner, 0],
        [outer, -inner, 0],
        [outer, -outer, 0],
        [inner, -outer, 0]]]
        )
    tx_back = np.array([[1, 0, 0, 0],
                        [0, 0,-1, -half_cube],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])
    tx_top = np.array([[1,0,0,0],
                       [0,1,0,0],
                       [0,0,1,half_cube],
                       [0,0,0,1]])
    tx_left = np.array([[ 0,0,-1, -half_cube],
                        [-1,0, 0, 0],
                        [ 0,1, 0, 0],
                        [ 0,0, 0, 1]])
    tx_right = np.array([[0,0,1,half_cube],
                         [1,0,0,0],
                         [0,1,0,0],
                         [0,0,0,1]])
    
    ids_to_3d = dict()
    for side, ids in side_to_ids.items():
        if side == 'back':
            tx = tx_back
        elif side == 'top':
            tx = tx_top
        elif side == 'left':
            tx = tx_left
        elif side == 'right':
            tx = tx_right
        else:
            assert False, f"Unknown side {side}"
        for i, id in enumerate(ids):
            points = np.array(face_coords[i])
            points = np.concatenate([points, np.ones((4,1))], axis=1)
            points = np.dot(tx, points.T).T
            points = points[:,:3]
            ids_to_3d[id] = points
    cube_params = {
        'cube_dim': aruco_cube_config_dict['cube_dim'],
        'ids_to_3d': ids_to_3d,
        'side_to_ids': aruco_cube_config_dict['cube_side2id_map'],
    }
    return cube_params

def get_aruco_dict(predefined:str
                   ) -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, predefined))


################# ArUco Functions #################
def detect_aruco_tags(img, aruco_dict, refine_subpix=True, estimate_pose=False, marker_size_map=None, mtx=None, dist=None):
    '''
    Detects ArUco tags in an image and estimating their pose.
    Input Parameters:
        img: np.ndarray: Image
        aruco_dict: cv2.aruco.Dictionary: ArUco dictionary
        refine_subpix: bool: Whether to refine subpixel
        estimate_pose: bool: Whether to estimate pose
        marker_size_map: dict: {tag_id: size} (required if estimate_pose=True)
        mtx: np.ndarray: Camera matrix (required if estimate_pose=True)
        dist: np.ndarray: Camera distortion coefficients (required if estimate_pose=True)
    '''
    aruco_params = cv2.aruco.DetectorParameters()    
    
    if refine_subpix:
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
        image=img, dictionary=aruco_dict, parameters=aruco_params)
    
    if len(corners)==0:
        return dict()
    
    if not estimate_pose:
        tag_dict = {int(this_id[0]): {'corners': this_corners.squeeze()} for this_id, this_corners in zip(ids, corners)}
        return tag_dict
    
    tag_dict = dict()
    for this_id, this_corners in zip(ids, corners):        
        this_id = int(this_id[0])
        if this_id not in marker_size_map:
            continue

        marker_size_m = marker_size_map[this_id]
        undistorted = cv2.undistortPoints(this_corners, mtx, dist, P=mtx)
        undistorted = np.expand_dims(undistorted.squeeze(), axis=0)
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
            undistorted, marker_size_m, mtx, dist)
        tag_dict[this_id] = {
            'rvec': rvec.squeeze(),
            'tvec': tvec.squeeze(),
            'corners': this_corners.squeeze()
        }
    return tag_dict

def estimate_aruco_cube_pose(tag_dict, mtx, dist, ids_to_3d):
    '''
    Estimate the pose of an ArUco cube.
    Input Parameters:
        tag_dict: dict: {tag_id: {'corners': np.ndarray}}
        ids_to_3d: dict: {tag_id: 3D points: np.ndarray}
    Returns:
        rvec: np.ndarray: Rotation vector
        tvec: np.ndarray: Translation vector
    '''
    if len(tag_dict) < 3:
        return None, None
    obj_points = []
    img_points = []
    for tag_id, tag in tag_dict.items():
        if tag_id not in ids_to_3d:
            continue
        obj_points.append(ids_to_3d[tag_id])
        img_points.append(tag['corners'])

    obj_points = np.array(obj_points, dtype=np.float32).reshape(1, -1, 3)
    img_points = np.array(img_points, dtype=np.float32).reshape(1, -1, 2)
    #ret_val, rvec, tvec, inliers = cv2.solvePnPRansac(obj_points, img_points, 
    #                                                  mtx, dist, reprojectionError=8.0) 
    ret_val, rvec, tvec = cv2.solvePnP(obj_points, img_points, mtx, dist)  

    if not ret_val:
        return None, None
    assert np.linalg.det(cv2.Rodrigues(rvec)[0]) > 0, "Rotation matrix is not right-handed"
    return rvec, tvec

def generate_board(ids, 
                   dimensions=(2, 2),
                   marker_length=2.5, 
                   marker_separation=1.0, 
                   side_padding=0.5,
                   cm2pixel_scaling=50):
    '''
    ids: list: List of ArUco marker IDs. The order is zigzag from top-left to bottom-right.
    cm2pixel_scaling: int: Number of pixels per cm
    marker_length: float: Marker length in cm
    marker_separation: float: Marker separation in cm
    side_padding: float: Side padding in cm
    '''

    # Create a grid of ArUco markers
    num_rows, num_cols = dimensions

    # Define the dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Define the board
    board = cv2.aruco.GridBoard(
        (num_cols, num_rows), marker_length, marker_separation, aruco_dict, ids=np.array(ids)
    )

    # Calculate the total board size
    total_width = num_cols * marker_length + (num_cols - 1) * marker_separation + 2 * side_padding
    total_height = num_rows * marker_length + (num_rows - 1) * marker_separation + 2 * side_padding

    # Create an image to draw the board
    image_size = int(max(total_width, total_height) * cm2pixel_scaling)  # Convert cm to pixels (assuming 100 pixels per cm)
    board_image = np.zeros((image_size, image_size), dtype=np.uint8)

    # Draw the board
    board_image = cv2.aruco.Board.generateImage(board, (image_size, image_size))

    # Add padding around the board
    padding_size = int(side_padding * cm2pixel_scaling)
    padded_image = cv2.copyMakeBorder(
        board_image,
        top=padding_size,
        bottom=padding_size,
        left=padding_size,
        right=padding_size,
        borderType=cv2.BORDER_CONSTANT,
        value=255  # White padding
    )
    return padded_image

def robust_filter(img, aruco_dict, board_img_dict, cube_len, rvec, tvec, mtx, dist):
    half_len = cube_len / 2.0
    # Define 3D points for the corners of each face of the cube
    faces_3d = {
        'top': np.array([[-half_len,  half_len, half_len],
                         [ half_len,  half_len, half_len],
                         [-half_len, -half_len, half_len],
                         [ half_len, -half_len, half_len]], dtype=np.float32),
                
        'back': np.array([[-half_len, -half_len,  half_len],
                          [ half_len, -half_len,  half_len],
                          [-half_len, -half_len, -half_len],
                          [ half_len, -half_len, -half_len]], dtype=np.float32),
        
        'left': np.array([[-half_len,  half_len,  half_len],
                          [-half_len, -half_len,  half_len],
                          [-half_len,  half_len, -half_len],
                          [-half_len, -half_len, -half_len]], dtype=np.float32),
        
        'right': np.array([[half_len, -half_len,  half_len],
                           [half_len,  half_len,  half_len],
                           [half_len, -half_len, -half_len],
                           [half_len,  half_len, -half_len]], dtype=np.float32)
    }
    face_normals = {
        'top': np.array([0, 0, 1], dtype=np.float32),
        'back': np.array([0, -1, 0], dtype=np.float32),
        'left': np.array([-1, 0, 0], dtype=np.float32),
        'right': np.array([1, 0, 0], dtype=np.float32)
    }
    # Determine visible sides
    R, _ = cv2.Rodrigues(rvec) 
    camera_position = (-R.T@tvec).squeeze()
    camera_normal = camera_position / np.linalg.norm(camera_position)
    # Normal vectors of each face in the cube's local coordinate system
    tag_dict = {}
    for face, points in faces_3d.items():
        if np.dot(face_normals[face], camera_normal) > 0:
            img_points, _ = cv2.projectPoints(points, rvec, tvec, mtx, dist)
            img_points = img_points.squeeze().astype(np.int32)
            # Change the perspective of the cube side to the board_image size
            board_img = board_img_dict[face]
            src_points = np.array([img_points[0], img_points[1], img_points[2], img_points[3]], dtype=np.float32)
            dst_points = np.array([[0, 0], [board_img.shape[1], 0], [0, board_img.shape[0]], [board_img.shape[1], board_img.shape[0]]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            warped_board = cv2.warpPerspective(img, M, (board_img.shape[1], board_img.shape[0]))

            # Apply adaptive thresholding to the warped board
            gray = cv2.cvtColor(warped_board, cv2.COLOR_BGR2GRAY)
            _, warped_board_threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Compare grayscale images warped_board_threshold and board_img_dict[face] similarity
            # TODO: This is slow, maybe find another way to compare images
            score = ssim(warped_board_threshold, board_img_dict[face])
            if score > 0.5:
                # Try detecting aruco markers on the warped board
                detected_tags = detect_aruco_tags(warped_board_threshold, aruco_dict)
                #warped_board = draw_aruco_corners(warped_board, detected_tags) # DEBUGGING PURPOSES
                # Undo perspective transform
                inv_M = np.linalg.inv(M)
                for tag_id, tag in detected_tags.items():
                    tag['corners'] = cv2.perspectiveTransform(np.array([tag['corners']]), inv_M)[0]
                tag_dict.update(detected_tags)
                #cv2.imwrite(f'data_aruco/output/valid/warped_board_{t_str}.jpg', warped_board) # DEBUGGING PURPOSES
            #else:
                # Save the warped board image for debugging
                # Include time in filename so it appears in chronological order
                #cv2.imwrite(f'data_aruco/output/invalid/warped_board_{t_str}.jpg', warped_board) # DEBUGGING PURPOSES
    return tag_dict     

################# Drawing Functions #################

def draw_aruco_corners(img, tag_dict):
    '''
    Draw detected ArUco tags and their axes on an image.
    Input Parameters:
        img: np.ndarray: Image
        tag_dict: dict: {tag_id: {'rvec': np.ndarray, 'tvec': np.ndarray, 'corners': np.ndarray}}
        mtx: np.ndarray: Camera matrix
        dist: np.ndarray: Camera distortion coefficients
        marker_size_map: dict
    '''
    if len(tag_dict) > 0:
        corners = [np.expand_dims(tag['corners'].squeeze(), axis=0) for tag in tag_dict.values()]
        img = cv2.aruco.drawDetectedMarkers(img, corners)
    return img

def draw_aruco_frame(img, tag_dict, mtx, dist, marker_size_map):
    '''
    Draw detected ArUco tags and their axes on an image.
    Input Parameters:
        img: np.ndarray: Image
        tag_dict: dict: {tag_id: {'rvec': np.ndarray, 'tvec': np.ndarray, 'corners': np.ndarray}}
        mtx: np.ndarray: Camera matrix
        dist: np.ndarray: Camera distortion coefficients
        marker_size_map: dict
    '''
    if len(tag_dict) > 0:
        for tag_id in tag_dict:
            img = cv2.drawFrameAxes(img, mtx, dist, tag_dict[tag_id]['rvec'], tag_dict[tag_id]['tvec'], marker_size_map[tag_id])
    return img

def draw_cube(img, cube_len, rvec, tvec, mtx, dist):
    '''
    Project a cube on an image.
    Input Parameters:
        img: np.ndarray: Image
        cube_len: float: Cube length
        rvec: np.ndarray: Rotation vector
        tvec: np.ndarray: Translation vector
        mtx: np.ndarray: Camera matrix
        dist: np.ndarray: Camera distortion coefficients
    Returns:
        img: np.ndarray: Image with cube projected on it
    '''
    half_len = cube_len/2.0
    axis_points = np.array([[[-half_len, -half_len, -half_len], 
                             [ half_len, -half_len, -half_len], 
                             [ half_len,  half_len, -half_len], 
                             [-half_len,  half_len, -half_len],
                             [-half_len, -half_len,  half_len], 
                             [ half_len, -half_len,  half_len], 
                             [ half_len,  half_len,  half_len], 
                             [-half_len,  half_len,  half_len]]])
    axis_points = axis_points.reshape(-1, 3)
    axis_points = np.array(axis_points, dtype=np.float32).reshape(1, -1, 3)
    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, mtx, dist)
    img_points = img_points.squeeze().astype(np.int32)
    img = cv2.polylines(img, [img_points[:4]], True, (0, 255, 0), 2)
    # Draw pillars
    for i in range(4):
        img = cv2.line(img, tuple(img_points[i]), tuple(img_points[i+4]), (0, 255, 0), 2)
    # Draw top
    img = cv2.polylines(img, [img_points[4:]], True, (0, 255, 0), 2)
    return img


def draw_aruco_board(img, board_img_dict, cube_len, rvec, tvec, mtx, dist):
    '''
    Project a cube on an image.
    Input Parameters:
        img: np.ndarray: Image
        board_img_dict: dict: Dictionary of board images for each face
        cube_len: float: Cube length
        rvec: np.ndarray: Rotation vector
        tvec: np.ndarray: Translation vector
        mtx: np.ndarray: Camera matrix
        dist: np.ndarray: Camera distortion coefficients
    Returns:
        img: np.ndarray: Image with cube projected on it
    '''
    half_len = cube_len / 2.0
    # Define 3D points for the corners of each face of the cube
    faces = {
        'top': np.array([[-half_len, half_len, half_len],
                         [half_len, half_len, half_len],
                         [-half_len, -half_len, half_len],
                         [half_len, -half_len, half_len]], dtype=np.float32),
                
        'back': np.array([[-half_len, -half_len, half_len],
                          [half_len, -half_len, half_len],
                          [-half_len, -half_len, -half_len],
                          [half_len, -half_len, -half_len]], dtype=np.float32),
        
        'left': np.array([[-half_len, half_len, half_len],
                          [-half_len, -half_len, half_len],
                          [-half_len, half_len, -half_len],
                          [-half_len, -half_len, -half_len]], dtype=np.float32),
        
        'right': np.array([[half_len, -half_len, half_len],
                           [half_len, half_len, half_len],
                           [half_len, -half_len, -half_len],
                           [half_len, half_len, -half_len]], dtype=np.float32)
    }

    # Normal vectors of each face in the cube's local coordinate system
    face_normals = {
        'top': np.array([0, 0, 1], dtype=np.float32),
        'back': np.array([0, -1, 0], dtype=np.float32),
        'left': np.array([-1, 0, 0], dtype=np.float32),
        'right': np.array([1, 0, 0], dtype=np.float32)
    }

    # Calculate the camera position in the cube's coordinate system
    R, _ = cv2.Rodrigues(rvec) 
    camera_position = (-R.T@tvec).squeeze()
    camera_normal = camera_position / np.linalg.norm(camera_position)
    for face, points in faces.items():
        if np.dot(face_normals[face], camera_normal) > 0:
            img_points, _ = cv2.projectPoints(points, rvec, tvec, mtx, dist)
            img_points = img_points.squeeze().astype(np.int32)

            # Get the corresponding board image for the face
            if face in board_img_dict:
                board_img = board_img_dict[face]

                # Define the source and destination points for perspective transform
                src_points = np.array([[0, 0], [board_img.shape[1], 0], [0, board_img.shape[0]], [board_img.shape[1], board_img.shape[0]]], dtype=np.float32)
                dst_points = np.array([img_points[0], img_points[1], img_points[2], img_points[3]], dtype=np.float32)

                # Compute the perspective transform matrix
                M = cv2.getPerspectiveTransform(src_points, dst_points)

                # Warp the board image onto the img
                warped_board = cv2.warpPerspective(board_img, M, (img.shape[1], img.shape[0]))

                # warped_board is 1 channel, img is 3 channel
                # Create a mask from the warped board
                warped_board_color = cv2.cvtColor(warped_board, cv2.COLOR_GRAY2BGR)

                # Create a mask from the warped board
                gray_board = cv2.cvtColor(warped_board_color, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_board, 1, 255, cv2.THRESH_BINARY)

                # Inverse mask to black out the area where the board will be placed
                mask_inv = cv2.bitwise_not(mask)
                img_bg = cv2.bitwise_and(img, img, mask=mask_inv)
                board_fg = cv2.bitwise_and(warped_board_color, warped_board_color, mask=mask)

                # Add the board foreground to the img background
                img = cv2.add(img_bg, board_fg)
    return img

################# Utility Functions #################

def average_rotations(rvecs):
    # Convert all rvecs to rotation matrices
    rotation_matrices = [cv2.Rodrigues(rvec)[0] for rvec in rvecs]
    
    # Compute the average of the rotation matrices
    mean_rotation_matrix = np.mean(rotation_matrices, axis=0)
    
    # Ensure the mean rotation matrix is orthogonal (re-orthogonalize it)
    U, _, Vt = np.linalg.svd(mean_rotation_matrix)
    mean_rotation_matrix = np.dot(U, Vt)
    
    # Convert the mean rotation matrix back to a rotation vector
    mean_rvec, _ = cv2.Rodrigues(mean_rotation_matrix)
    
    return mean_rvec

################# Video Processing Functions #################

def extract_aruco_cube_pose_from_video(video_path, aruco_dict, mtx, dist, cube_params):
    '''
    Extract the ArUco cube pose from a video.
    Input Parameters:
        video_path: str: Path to the video
        aruco_dict: cv2.aruco.Dictionary: ArUco dictionary
        marker_size_map: dict: {tag_id: size}
        mtx: np.ndarray: Camera matrix
        dist: np.ndarray: Camera distortion coefficients
        cube_params: dict: {'cube_dim': float, tag_id: str}
    Returns:
        cube_poses: dict: {'timestamps': np.ndarray, 'rvecs': np.ndarray, 'tvecs': np.ndarray}
    '''
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    
    board_img_dict = {
        'top': generate_board(cube_params['side_to_ids']['top']),
        'back': generate_board(cube_params['side_to_ids']['back']),
        'left': generate_board(cube_params['side_to_ids']['left']),
        'right': generate_board(cube_params['side_to_ids']['right'])
    }

    expected_img_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cube_poses = {
        'timestamps': [],
        'rvecs': [],
        'tvecs': []
    }
    frame_idx = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        tag_dict = detect_aruco_tags(img, aruco_dict)
        rvec, tvec = estimate_aruco_cube_pose(tag_dict, mtx, dist, cube_params['ids_to_3d'])
        if rvec is not None and tvec is not None:
            tag_dict = robust_filter(img, aruco_dict, board_img_dict, cube_params['cube_dim'], rvec, tvec, mtx, dist)
            rvec, tvec = estimate_aruco_cube_pose(tag_dict, mtx, dist, cube_params['ids_to_3d'])
        if rvec is not None and tvec is not None:
            cube_poses['timestamps'].append(cap.get(cv2.CAP_PROP_POS_MSEC)/S_TO_MS)
            cube_poses['rvecs'].append(rvec)
            cube_poses['tvecs'].append(tvec)
        frame_idx += 1
    # Convert lists to numpy arrays
    cube_poses['timestamps'] = np.array(cube_poses['timestamps'])
    cube_poses['rvecs'] = np.array(cube_poses['rvecs']).squeeze()
    cube_poses['tvecs'] = np.array(cube_poses['tvecs'])
    print(f"Expected img count: {expected_img_count}, Actual img count: {frame_idx}")
    return cube_poses

def example_record_aruco_video(video_path, aruco_dict, mtx, dist, cube_params, output_path):
    '''
    Record a video with the ArUco cube pose overlayed on each frame.
    Input Parameters:
        video_path: str: Path to the video
        aruco_dict: cv2.aruco.Dictionary: ArUco dictionary
        marker_size_map: dict: {tag_id: size}
        mtx: np.ndarray: Camera matrix
        dist: np.ndarray: Camera distortion coefficients
        cube_params: dict: {'cube_dim': float, tag_id: str}
    '''
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    expected_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_frame_count = 0
    num_detected_cube = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Get frame rate of video and frame width and height
    frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

    board_img_dict = {
        'top': generate_board(cube_params['side_to_ids']['top']),
        'back': generate_board(cube_params['side_to_ids']['back']),
        'left': generate_board(cube_params['side_to_ids']['left']),
        'right': generate_board(cube_params['side_to_ids']['right'])
    }

    while True:
        t = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        else:
            actual_frame_count+=1
        tag_dict = detect_aruco_tags(frame, aruco_dict)
        rvec, tvec = estimate_aruco_cube_pose(tag_dict, mtx, dist, cube_params['ids_to_3d'])
        if rvec is not None and tvec is not None:
            tag_dict = robust_filter(frame, aruco_dict, board_img_dict, cube_params['cube_dim'], rvec, tvec, mtx, dist)
            rvec, tvec = estimate_aruco_cube_pose(tag_dict, mtx, dist, cube_params['ids_to_3d'])
        if rvec is not None and tvec is not None:
            frame = cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.1)
            frame = draw_aruco_corners(frame, tag_dict)
            frame = draw_cube(frame, cube_params['cube_dim'], rvec, tvec, mtx, dist)
            num_detected_cube += 1
        out.write(frame)
        print(f'Processed frame: {actual_frame_count} / {expected_frame_count} | Processing rate: {1/(time.time()-t):.2f} fps')
    cap.release()
    out.release()
    print(f"Output video saved as {output_path}. Expected frame count: {expected_frame_count}, Actual frame count: {actual_frame_count}")
    print(f"Number of detected cubes: {num_detected_cube}/{actual_frame_count}")

