'''
Converted from C++ to Python by John Kim and ChatGPT4o
Original C++ code from https://github.com/madcowswe/ESKF/blob/master/src/main.cpp
NOTE:
scipy quaternion: [x, y, z, w]
code uses hamilton quaternion: [w, x, y, z]
'''

import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from unrolled_FPFt import unrolledFPFt
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
from utils import gopro_utils
from utils import interpolation_utils 
import click

# Transformation of camera w.r.t IMU (Or maybe other way around)
#TX_IMU_CAMERA = np.array([[-1, 0, 0, 0.003],
#                         [ 0, 0,-1,-0.03 ],
#                         [ 0,-1, 0, 0    ],
#                         [ 0, 0, 0, 1    ]])
# NOTE: not using the above because unsure which way it is. For safe guess, assume all 0. 
# Transformation of camera w.r.t IMU (Or maybe other way around)
TX_IMU_CAMERA = np.array([[-1, 0, 0,  0],
                          [ 0, 0,-1,  0],
                          [ 0,-1, 0,  0],
                          [ 0, 0, 0,  1]])

GRAVITY = 9.81

POS_IDX = 0
VEL_IDX = POS_IDX + 3
QUAT_IDX = VEL_IDX + 3
AB_IDX = QUAT_IDX + 4
GB_IDX = AB_IDX + 3
STATE_SIZE = GB_IDX + 3

dPOS_IDX = 0
dVEL_IDX = dPOS_IDX + 3
dTHETA_IDX = dVEL_IDX + 3
dAB_IDX = dTHETA_IDX + 3
dGB_IDX = dAB_IDX + 3
dSTATE_SIZE = dGB_IDX + 3


SQ = lambda x: x * x
I_3 = np.eye(3)
I_dx = np.eye(dSTATE_SIZE)

class ESKF:
    def __init__(self, 
                 sigma_accel, # [m/s^2]
                 sigma_gyro, # [rad/s]
                 sigma_init_pos, # [m]
                 sigma_init_vel, # [m/s]
                 sigma_init_dtheta, # [rad]
                 mode # 'gripper' or 'pruner'
                 ):
        
        # NOTE: TX_CUBE_IMU: Frame of IMU with respect to the CUBE frame
        # Gripper
        if mode=='gripper':
            self.TX_CUBE_IMU = np.array([[-1, 0, 0, 0.00 ],
                                         [ 0,-1, 0, 0.065],
                                         [ 0, 0, 1, 0.05 ],
                                         [ 0, 0, 0, 1    ]])
        elif mode=='pruner':
            self.TX_CUBE_IMU = np.array([[-1, 0, 0, 0.00 ],
                                         [ 0,-1, 0, 0.105],
                                         [ 0, 0, 1, 0.04 ],
                                         [ 0, 0, 0, 1    ]])
        elif mode=='pruner_inverse':
            self.TX_CUBE_IMU = np.array([[ 1, 0, 0, 0.00 ],
                                         [ 0,-1, 0, 0.085],
                                         [ 0, 0,-1,-0.207 ],
                                         [ 0, 0, 0, 1    ]])
        
        sigma_accel_drift = 0.001 * sigma_accel  # [m/s^2 sqrt(s)] (Educated guess, real value to be measured)
        sigma_gyro_drift = 0.001 * sigma_gyro  # [rad/s sqrt(s)] (Educated guess, real value to be measured)
        sigma_init_accel_bias = 10000 * sigma_accel_drift  # [m/s^2]
        sigma_init_gyro_bias = 10000 * sigma_gyro_drift  # [rad/s]

        ESKF.make_P(
            SQ(sigma_init_pos) * I_3,
            SQ(sigma_init_vel) * I_3,
            SQ(sigma_init_dtheta) * I_3,
            SQ(sigma_init_accel_bias) * I_3,
            SQ(sigma_init_gyro_bias) * I_3
        )

        self.var_acc_ = SQ(sigma_accel)
        self.var_omega_ = SQ(sigma_gyro)
        self.var_acc_bias_ = SQ(sigma_accel_drift)
        self.var_omega_bias_ = SQ(sigma_gyro_drift)
        self.a_gravity_ = np.array([0, GRAVITY, 0])

        self.nominalState_ = self.make_state(
            np.array([0, 0, 0]),  # init pos
            np.array([0, 0, 0]),  # init vel
            np.array([1, 0, 0, 0]),  # init hamiltonion quaternion wxyz 
            np.array([0, 0, 0]),  # init accel bias
            np.array([0, 0, 0])  # init gyro bias
        )

        self.P_ = self.make_P(
            SQ(sigma_init_pos) * I_3,
            SQ(sigma_init_vel) * I_3,
            SQ(sigma_init_dtheta) * I_3,
            SQ(sigma_init_accel_bias) * I_3,
            SQ(sigma_init_gyro_bias) * I_3
        )

        # Jacobian of the state transition: page 59, eqn 269
        # Precompute constant part only
        self.F_x_ = np.zeros((dSTATE_SIZE, dSTATE_SIZE))
        # dPos row
        self.F_x_[dPOS_IDX:dPOS_IDX+3, dPOS_IDX:dPOS_IDX+3] = I_3
        # dVel row
        self.F_x_[dVEL_IDX:dVEL_IDX+3, dVEL_IDX:dVEL_IDX+3] = I_3
        # dAccelBias row
        self.F_x_[dAB_IDX:dAB_IDX+3, dAB_IDX:dAB_IDX+3] = I_3
        # dGyroBias row
        self.F_x_[dGB_IDX:dGB_IDX+3, dGB_IDX:dGB_IDX+3] = I_3
    
    @staticmethod
    def make_state(p, v, q, a_b, omega_b):
        '''
        x, y, z: position
        vx, vy, vz: velocity
        qw, qx, qy, qz: hamilton quaternion
        ax, ay, az: accelerometer bias
        gx, gy, gz: gyro bias
        '''
        out = np.zeros((STATE_SIZE, 1))
        out[:3, 0] = p
        out[3:6, 0] = v
        out[6:10, 0] = q/np.linalg.norm(q) # normalize quaternion
        out[10:13, 0] = a_b
        out[13:16, 0] = omega_b
        return out

    @staticmethod
    def make_P(cov_pos, cov_vel, cov_dtheta, cov_a_b, cov_omega_b):
        P = np.zeros((dSTATE_SIZE, dSTATE_SIZE))
        P[dPOS_IDX:dPOS_IDX+3, dPOS_IDX:dPOS_IDX+3] = cov_pos
        P[dVEL_IDX:dVEL_IDX+3, dVEL_IDX:dVEL_IDX+3] = cov_vel
        P[dTHETA_IDX:dTHETA_IDX+3, dTHETA_IDX:dTHETA_IDX+3] = cov_dtheta
        P[dAB_IDX:dAB_IDX+3, dAB_IDX:dAB_IDX+3] = cov_a_b
        P[dGB_IDX:dGB_IDX+3, dGB_IDX:dGB_IDX+3] = cov_omega_b
        return P

    ######################## GETTERS ########################
    def get_pos(self):
        return self.nominalState_[POS_IDX:POS_IDX + 3].flatten()

    def get_vel(self):
        return self.nominalState_[VEL_IDX:VEL_IDX + 3].flatten()

    def get_quat_wxyz(self):
        return self.nominalState_[QUAT_IDX:QUAT_IDX + 4].flatten()

    def get_quat_xyzw(self):
        return self.wxyz_to_xyzw(self.get_quat_wxyz()).flatten()

    def get_accel_bias(self):
        return self.nominalState_[AB_IDX:AB_IDX + 3].flatten()

    def get_gyro_bias(self):
        return self.nominalState_[GB_IDX:GB_IDX + 3].flatten()

    def get_matrix(self):
        return R.from_quat(self.get_quat_xyzw()).as_matrix()

    ######################## STATIC HELPERS ########################
    @staticmethod
    def wxyz_to_xyzw(q):
        '''
        Convert hamilton quaternion (w, x, y, z) to scipy quaternion (x, y, z, w)
        '''
        return np.array([q[1], q[2], q[3], q[0]]).flatten()

    @staticmethod
    def xyzw_to_wxyz(q):
        '''
        Convert scipy quaternion (x, y, z, w) to hamilton quaternion (w, x, y, z)
        '''
        return np.array([q[3], q[0], q[1], q[2]]).flatten()

    @staticmethod
    def get_skew(vec):
        return np.array([
            [0, -vec[2], vec[1]],
            [vec[2], 0, -vec[0]],
            [-vec[1], vec[0], 0]
        ])

    ######################## MAIN ESKF METHODS ########################
    def predict_imu(self, a_m, omega_m, dt):
        # Rotation matrix of current state
        Rot = self.get_matrix()
        # Accelerometer measurement
        acc_body = a_m - self.get_accel_bias()
        acc_global = Rot @ acc_body
        # Gyro measurement
        omega = omega_m - self.get_gyro_bias()
        delta_theta = omega * dt
        q_delta_theta = R.from_rotvec(delta_theta)
        R_delta_theta = q_delta_theta.as_matrix()

        # Nominal state kinematics (eqn 259, pg 58)
        delta_pos = self.get_vel() * dt + 0.5 * (acc_global + self.a_gravity_) * dt * dt
        self.nominalState_[POS_IDX:POS_IDX+3, 0] += delta_pos
        self.nominalState_[VEL_IDX:VEL_IDX+3, 0] += (acc_global + self.a_gravity_) * dt
        q_state = R.from_quat(self.get_quat_xyzw())
        q_predict = (q_state * q_delta_theta).as_quat().flatten()
        self.nominalState_[QUAT_IDX:QUAT_IDX+4, 0] = self.xyzw_to_wxyz(q_predict) 
        
        # Predict P and inject variance (with diagonal optimization)
        Pnew = np.zeros_like(self.P_)
        unrolledFPFt(self.P_, Pnew, dt,
                     -Rot @ self.get_skew(acc_body) * dt,
                     -Rot * dt,
                     R_delta_theta.T)
        self.P_ = Pnew

        # Inject process noise
        self.P_[dVEL_IDX:dVEL_IDX+3, dVEL_IDX:dVEL_IDX+3] += self.var_acc_ * SQ(dt) * I_3
        self.P_[dTHETA_IDX:dTHETA_IDX+3, dTHETA_IDX:dTHETA_IDX+3] += self.var_omega_ * SQ(dt) * I_3
        self.P_[dAB_IDX:dAB_IDX+3, dAB_IDX:dAB_IDX+3] += self.var_acc_bias_ * dt * I_3
        self.P_[dGB_IDX:dGB_IDX+3, dGB_IDX:dGB_IDX+3] += self.var_omega_bias_ * dt * I_3

    def measure_pos(self, pos_meas, pos_covariance):
        delta_pos = pos_meas - self.get_pos()
        H = np.zeros((3, dSTATE_SIZE))
        H[:, dPOS_IDX:dPOS_IDX+3] = I_3
        self.update_3D(delta_pos, pos_covariance, H)

    def measure_quat(self, q_gb_meas, theta_covariance):
        '''
        q_gb_meas: quaternion from global to body frame [w, x, y, z]
        '''
        q_gb_nominal = self.get_quat_xyzw()
        q_gb_nominal = R.from_quat(q_gb_nominal)
        q_gb_meas = self.wxyz_to_xyzw(q_gb_meas)
        q_gb_meas = R.from_quat(q_gb_meas)
        q_bNominal_bMeas = (q_gb_nominal.inv() * q_gb_meas).as_quat().flatten()
        delta_theta = R.from_quat(q_bNominal_bMeas).as_rotvec()
        H = np.zeros((3, dSTATE_SIZE))
        H[:, dTHETA_IDX:dTHETA_IDX+3] = I_3
        self.update_3D(delta_theta, theta_covariance, H)

    def update_3D(self, delta_measurement, meas_covariance, H):
        PHt = self.P_ @ H.T
        K = PHt @ np.linalg.inv(H @ PHt + meas_covariance)
        errorState = K @ delta_measurement
        I_KH = I_dx - K @ H
        self.P_ = I_KH @ self.P_ @ I_KH.T + K @ meas_covariance @ K.T
        self.inject_error_state(errorState)

    def inject_error_state(self, error_state):
        self.nominalState_[POS_IDX:POS_IDX+3, 0] += error_state[dPOS_IDX:dPOS_IDX+3]
        self.nominalState_[VEL_IDX:VEL_IDX+3, 0] += error_state[dVEL_IDX:dVEL_IDX+3]
        dtheta = error_state[dTHETA_IDX:dTHETA_IDX+3]
        q_dtheta = R.from_rotvec(dtheta)
        q_state = R.from_quat(self.get_quat_xyzw())
        q_update = (q_state * q_dtheta).as_quat().flatten()
        self.nominalState_[QUAT_IDX:QUAT_IDX+4, 0] = self.xyzw_to_wxyz(q_update)
        self.nominalState_[AB_IDX:AB_IDX+3, 0] += error_state[dAB_IDX:dAB_IDX+3]
        self.nominalState_[GB_IDX:GB_IDX+3, 0] += error_state[dGB_IDX:dGB_IDX+3]

        G_theta = I_3 - self.get_skew(0.5 * dtheta)
        self.P_[dTHETA_IDX:dTHETA_IDX+3, dTHETA_IDX:dTHETA_IDX+3] = G_theta @ self.P_[dTHETA_IDX:dTHETA_IDX+3, dTHETA_IDX:dTHETA_IDX+3] @ G_theta.T

    def run(self):
        sigma_aruco_pos = 0.003  # [m]
        sigma_aruco_rot = 0.03  # [rad]

        # Run main loop of the EKF
        ekf_all_states = []
        ekf_all_states_t = []
        ekf_update_states = []
        ekf_update_states_t = []
        measured_states = []
        measured_states_t = []
        imu_index, pose_index = 0, 0
        while imu_index<len(self.imu_data) and pose_index<len(self.pose_data):
            # PREDICT the next state using the IMU data
            if self.imu_data[imu_index, 0] < self.pose_data[pose_index, 0]:
                timestamp = self.imu_data[imu_index, 0]
                if imu_index==0:
                    dt = 0
                else:
                    dt = self.imu_data[imu_index, 0] - self.imu_data[imu_index-1, 0]
                acc = self.imu_data[imu_index, 1:4]
                gyro = self.imu_data[imu_index, 4:7]
                self.predict_imu(acc, gyro, dt)
                imu_index += 1
            # UPDATE the state using the ArUco pose data
            else:
                timestamp = self.pose_data[pose_index, 0]
                tvec = self.pose_data[pose_index, 1:4]
                rvec = self.pose_data[pose_index, 4:7]

                q_meas = R.from_rotvec(rvec).as_quat()
                q_meas = self.xyzw_to_wxyz(q_meas)
                self.measure_pos(tvec, SQ(sigma_aruco_pos) * I_3)
                self.measure_quat(q_meas, SQ(sigma_aruco_rot) * I_3)

                pose_index += 1
                ekf_update_states.append([*self.get_pos(), *self.get_quat_xyzw()])
                ekf_update_states_t.append(timestamp)
                measured_states.append([*tvec, *self.wxyz_to_xyzw(q_meas)])
                measured_states_t.append(timestamp)
            ekf_all_states.append([*self.get_pos(), *self.get_quat_xyzw()])
            ekf_all_states_t.append(timestamp)

        self.ekf_all_states = np.array(ekf_all_states)
        self.ekf_all_states_t = np.array(ekf_all_states_t)
        self.ekf_update_states = np.array(ekf_update_states)
        self.ekf_update_states_t = np.array(ekf_update_states_t)
        self.measured_states = np.array(measured_states)
        self.measured_states_t = np.array(measured_states_t)

    ######################## IMPORT DATA ########################
    def import_data(self, data_path, visualize=False):
        self.data_path = data_path
        be_video_path = os.path.join(data_path, 'birdseye_video.mp4')
        raw_video_path = os.path.join(data_path, 'raw_video.mp4')
        be_meta = gopro_utils.get_metadata(be_video_path)
        raw_meta = gopro_utils.get_metadata(raw_video_path)
        be_start, be_end = be_meta['start_timestamp'], be_meta['end_timestamp']
        raw_start, raw_end = raw_meta['start_timestamp'], raw_meta['end_timestamp']
        # Need to adjust the start and end timestamps to match the overlap between the two videos
        offset = be_start - raw_start - 0.08 

        imu_json_path = os.path.join(data_path, 'imu_data.json')
        cube_pose_pkl_path = os.path.join(data_path, 'cube_detection.pkl')

        # Load telemetry data
        vTimeStamps = []
        coriTimeStamps = []
        vAcc = []
        vGyro = []
        success = gopro_utils.load_telemetry(imu_json_path, vTimeStamps, coriTimeStamps, vAcc, vGyro)
        assert success, "Failed to load telemetry"
        vTimeStamps = np.array(vTimeStamps)
        coriTimeStamps = np.array(coriTimeStamps)
        imu_data = []
        for t, acc, gyro in zip(vTimeStamps, vAcc, vGyro):
            imu_data.append([t, *acc, *gyro])
        imu_data = np.array(imu_data)
        self.imu_data = imu_data
        self.imu_frequency = 1 / np.mean(np.diff(imu_data[:, 0]))
        # Get aruco cube poses by loading from pickle file
        poses = pickle.load(open(cube_pose_pkl_path, 'rb'))
        cube_timestamps = poses['timestamps']
        cube_rvecs = poses['rvecs']
        cube_tvecs = poses['tvecs']
        pose_data = []
        for t, rvec, tvec in zip(cube_timestamps, cube_rvecs, cube_tvecs):
            r = R.from_rotvec(rvec)
            tx_becamera_cube = np.eye(4)
            tx_becamera_cube[:3, :3] = r.as_matrix()
            tx_becamera_cube[:3, 3] = tvec.flatten()
            tx_becamera_imu = tx_becamera_cube@self.TX_CUBE_IMU
            tvec = tx_becamera_imu[:3, 3]
            rvec = R.from_matrix(tx_becamera_imu[:3, :3]).as_rotvec()
            pose_data.append([t, *tvec, *rvec])
        pose_data = np.array(pose_data)
        pose_data = self._pose_measurement_filter(pose_data)
        # Offset the timestamps of the pose data to match the IMU data
        pose_data[:, 0] += offset
        # Remove poses that are before IMU data
        self.pose_data = pose_data[pose_data[:, 0] > 0.0]

        # Initialize the state with the first pose data
        init_pos = self.pose_data[0, 1:4]
        init_quat = R.from_rotvec(self.pose_data[0, 4:7]).as_quat()
        init_quat = self.xyzw_to_wxyz(init_quat)
        self.nominalState_ = self.make_state(
            init_pos,  # init pos
            np.array([0, 0, 0]),  # init vel
            init_quat,  # init hamiltonion quaternion wxyz 
            np.array([0, 0, 0]),  # init accel bias
            np.array([0, 0, 0])  # init gyro bias
        )

        self.frame_frequency = float(raw_meta['fps'])
        print(f'Data imported successfully:\
              \nIMU.shape: {imu_data.shape} @ {self.imu_frequency:.0f} Hz\
              \nPose.shape: {pose_data.shape} @ {self.frame_frequency:.0f} Hz\
              \nPose data offset by {offset:.4f} seconds')
        
        if visualize:
            plt.figure(figsize=(16, 10))
            # Determine the common range of timestamps
            min_time = min(imu_data[:, 0].min(), pose_data[:, 0].min())
            max_time = max(imu_data[:, 0].max(), pose_data[:, 0].max())
            # Plot the accelerometer data and gyroscope data, and pose_data side by side
            ax = plt.subplot(4, 1, 1)
            ax.plot(imu_data[:, 0], imu_data[:, 1], label='Acc X')
            ax.plot(imu_data[:, 0], imu_data[:, 2], label='Acc Y')
            ax.plot(imu_data[:, 0], imu_data[:, 3], label='Acc Z')
            ax.legend()
            ax.set_title('Accelerometer Data')
            ax.set_ylabel('Acceleration (m/s^2)')
            ax.set_xlim(min_time, max_time)

            ax = plt.subplot(4, 1, 2)
            ax.plot(imu_data[:, 0], imu_data[:, 4], label='Gyro X')
            ax.plot(imu_data[:, 0], imu_data[:, 5], label='Gyro Y')
            ax.plot(imu_data[:, 0], imu_data[:, 6], label='Gyro Z')
            ax.legend()
            ax.set_title('Gyroscope Data')
            ax.set_ylabel('Angular Velocity (rad/s)')
            ax.set_xlim(min_time, max_time)
            
            # Plot points instead of line for pose data
            ax = plt.subplot(4, 1, 3)
            ax.plot(pose_data[:, 0], pose_data[:, 1], '.', label='Pose X', markersize=2)
            ax.plot(pose_data[:, 0], pose_data[:, 2], '.', label='Pose Y', markersize=2)
            ax.plot(pose_data[:, 0], pose_data[:, 3], '.', label='Pose Z', markersize=2)
            ax.legend() 
            ax.set_title('Pose Data')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Position (m)')
            ax.set_xlim(min_time, max_time)

            # Plot points instead of line for pose data
            # Convert pose_data[:, 4:7] rotation vectors to euler angles 
            rpy = R.from_rotvec(pose_data[:, 4:7]).as_euler('xyz')
            ax = plt.subplot(4, 1, 4)
            ax.plot(pose_data[:, 0], rpy[:, 0], '.', label='Roll', markersize=2)
            ax.plot(pose_data[:, 0], rpy[:, 1], '.', label='Pitch', markersize=2)
            ax.plot(pose_data[:, 0], rpy[:, 2], '.', label='Yaw', markersize=2)
            ax.legend()
            ax.set_title('Pose Data')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Angle (rad)')
            ax.set_xlim(min_time, max_time)
            plt.savefig(os.path.join(self.data_path, '_ekf_raw_data.png'))

    @staticmethod
    def _pose_measurement_filter(pose_data):
        '''
        Given a moving window of size 3, compare the current pose data with the previous and next pose data.
        Compute the difference between previous and next pose data
        If the difference between current and prevs, and current and next is greater than n times the difference between prev and next, then remove the current pose data.
        '''
        n = 2
        i = 1
        while i < len(pose_data) - 1:
            diff = pose_data[i-1, 1:4] - pose_data[i+1, 1:4]
            prev_diff = pose_data[i-1, 1:4] - pose_data[i, 1:4]
            next_diff = pose_data[i+1, 1:4] - pose_data[i, 1:4]
            
            if np.linalg.norm(prev_diff) > n * np.linalg.norm(diff) or \
            np.linalg.norm(next_diff) > n * np.linalg.norm(diff) or \
            np.linalg.norm(pose_data[i, 1:4] - pose_data[i-1, 1:4]) / (pose_data[i, 0] - pose_data[i-1, 0]) > 1.0:
                pose_data = np.delete(pose_data, i, axis=0)
            else:
                i += 1
        return pose_data
    
    def convert_frame_from_imu_to_camera(self):
        # Currently, self.measured_states and self.ekf_all_states are in IMU frame
        # We want to convert them to the camera frame using the transformation matrix TX_IMU_CAMERA
        measured_states = np.zeros((self.measured_states.shape[0], 4, 4))
        ekf_all_states = np.zeros((self.ekf_all_states.shape[0], 4, 4))

        for i in range(self.measured_states.shape[0]):
            tx_imu = np.eye(4)
            tx_imu[:3, 3] = self.measured_states[i, :3]
            tx_imu[:3, :3] = R.from_quat(self.measured_states[i, 3:7]).as_matrix()
            measured_states[i] = tx_imu @ TX_IMU_CAMERA
            self.measured_states[i, :3] = measured_states[i][:3, 3]
            self.measured_states[i, 3:7] = R.from_matrix(measured_states[i][:3, :3]).as_quat()

        for i in range(self.ekf_all_states.shape[0]):
            tx_imu = np.eye(4)
            tx_imu[:3, 3] = self.ekf_all_states[i, :3]
            tx_imu[:3, :3] = R.from_quat(self.ekf_all_states[i, 3:7]).as_matrix()
            ekf_all_states[i] = tx_imu @ TX_IMU_CAMERA
            self.ekf_all_states[i, :3] = ekf_all_states[i][:3, 3]
            self.ekf_all_states[i, 3:7] = R.from_matrix(ekf_all_states[i][:3, :3]).as_quat()

    def plot_results(self, save=False):
        marker_size = 1
        
        ################## Plot 3D Trajectory and Position Profiles ##################
        fig = plt.figure(figsize=(12, 10))

        # Create 3D trajectory plot
        ax3 = fig.add_subplot(2, 2, 1, projection='3d')
        ax3.plot(self.measured_states[:, 0], self.measured_states[:, 1], self.measured_states[:, 2], 
            label=f'Aruco Trajectory ({self.frame_frequency:.0f} Hz)', marker='x', linestyle='')
        ax3.plot(self.ekf_all_states[:, 0], self.ekf_all_states[:, 1], self.ekf_all_states[:, 2], 
            label=f'KF Trajectory ({self.imu_frequency:.0f} Hz)', marker='.', linestyle='', markersize=marker_size)
        ax3.legend()
        ax3.set_title('3D Trajectory')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        # Make the aspect ratio true to scale
        ax3.set_box_aspect([1,1,1])
        # Set axis limits taking into account both KF and Aruco data
        min_x = min(np.min(self.ekf_all_states[:, 0]), np.min(self.measured_states[:, 0]))
        max_x = max(np.max(self.ekf_all_states[:, 0]), np.max(self.measured_states[:, 0]))
        min_y = min(np.min(self.ekf_all_states[:, 1]), np.min(self.measured_states[:, 1]))
        max_y = max(np.max(self.ekf_all_states[:, 1]), np.max(self.measured_states[:, 1]))
        min_z = min(np.min(self.ekf_all_states[:, 2]), np.min(self.measured_states[:, 2]))
        max_z = max(np.max(self.ekf_all_states[:, 2]), np.max(self.measured_states[:, 2]))

        # Set equal aspect ratio
        max_range = np.array([max_x-min_x, max_y-min_y, max_z-min_z]).max() / 2.0
        mid_x = (max_x + min_x) * 0.5
        mid_y = (max_y + min_y) * 0.5
        mid_z = (max_z + min_z) * 0.5
        ax3.set_xlim(mid_x - max_range, mid_x + max_range)
        ax3.set_ylim(mid_y - max_range, mid_y + max_range)
        ax3.set_zlim(mid_z - max_range, mid_z + max_range)

        # Draw a frame at the origin
        origin = np.eye(4)
        ax3.quiver(origin[0, 3], origin[1, 3], origin[2, 3], origin[0, 0], origin[1, 0], origin[2, 0], color='r')
        ax3.quiver(origin[0, 3], origin[1, 3], origin[2, 3], origin[0, 1], origin[1, 1], origin[2, 1], color='g')
        ax3.quiver(origin[0, 3], origin[1, 3], origin[2, 3], origin[0, 2], origin[1, 2], origin[2, 2], color='b')

        # Create 2D position profiles
        ax = fig.add_subplot(2, 2, 2)
        ax.plot(self.measured_states_t, self.measured_states[:, 0], 
            label=f'Aruco X ({self.frame_frequency:.0f} Hz)', marker='x', linestyle='')
        ax.plot(self.ekf_all_states_t, self.ekf_all_states[:, 0], 
            label=f'KF X ({self.imu_frequency:.0f} Hz)', marker='.', linestyle='', markersize=marker_size)
        ax.legend()
        ax.set_title('Position X')
        ax.set_ylabel('Position (m)')

        ax = fig.add_subplot(2, 2, 3)
        ax.plot(self.measured_states_t, self.measured_states[:, 1], 
            label=f'Aruco Y ({self.frame_frequency:.0f} Hz)', marker='x', linestyle='')
        ax.plot(self.ekf_all_states_t, self.ekf_all_states[:, 1], 
            label=f'KF Y ({self.imu_frequency:.0f} Hz)', marker='.', linestyle='', markersize=marker_size)
        ax.legend()
        ax.set_title('Position Y')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (m)')

        ax = fig.add_subplot(2, 2, 4)
        ax.plot(self.measured_states_t, self.measured_states[:, 2], 
            label=f'Aruco Z ({self.frame_frequency:.0f} Hz)', marker='x', linestyle='')
        ax.plot(self.ekf_all_states_t, self.ekf_all_states[:, 2], 
            label=f'KF Z ({self.imu_frequency:.0f} Hz)', marker='.', linestyle='', markersize=marker_size)
        ax.legend()
        ax.set_title('Position Z')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (m)')

        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.data_path, '_ekf_pos.png'))
            plt.close()
        else:
            plt.show()

        ################## Plot orientation angles ##################
        fig = plt.figure(figsize=(12, 10))
        ax = plt.subplot(3, 1, 1)
        rpy_meas = R.from_quat(self.measured_states[:, 3:7]).as_euler('xyz')
        rpy_kf = R.from_quat(self.ekf_all_states[:, 3:7]).as_euler('xyz')
        ax.plot(self.measured_states_t, rpy_meas[:, 0], 
            label=f'Aruco Roll ({self.frame_frequency:.0f} Hz)', marker='x', linestyle='')
        ax.plot(self.ekf_all_states_t, rpy_kf[:, 0], 
            label=f'KF Roll ({self.imu_frequency:.0f} Hz)', marker='.', linestyle='', markersize=marker_size)
        ax.legend()
        ax.set_title('Roll')
        ax.set_ylabel('Angle (rad)')
        ax = plt.subplot(3, 1, 2)
        ax.plot(self.measured_states_t, rpy_meas[:, 1], 
            label=f'Aruco Pitch ({self.frame_frequency:.0f} Hz)', marker='x', linestyle='')
        ax.plot(self.ekf_all_states_t, rpy_kf[:, 1], 
            label=f'KF Pitch ({self.imu_frequency:.0f} Hz)', marker='.', linestyle='', markersize=marker_size)
        ax.legend()
        ax.set_title('Pitch')
        ax.set_ylabel('Angle (rad)')
        ax = plt.subplot(3, 1, 3)
        ax.plot(self.measured_states_t, rpy_meas[:, 2], 
            label=f'Aruco Yaw ({self.frame_frequency:.0f} Hz)', marker='x', linestyle='')
        ax.plot(self.ekf_all_states_t, rpy_kf[:, 2], 
            label=f'KF Yaw ({self.imu_frequency:.0f} Hz)', marker='.', linestyle='', markersize=marker_size)
        ax.legend()
        ax.set_title('Yaw')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (rad)')
        if save:
            plt.savefig(os.path.join(self.data_path, '_ekf_ori.png'))
            plt.close()
        else:
            plt.show()

        # Plot pos, ori all on the same plot stacked vertically
        fig = plt.figure(figsize=(12, 20))
        # Create 2D position profiles
        ax = fig.add_subplot(6, 1, 1)
        ax.plot(self.measured_states_t, self.measured_states[:, 0], 
            label=f'Aruco X ({self.frame_frequency:.0f} Hz)', marker='x', linestyle='')
        ax.plot(self.ekf_all_states_t, self.ekf_all_states[:, 0], 
            label=f'KF X ({self.imu_frequency:.0f} Hz)', marker='.', linestyle='', markersize=marker_size)
        ax.legend()
        ax.set_ylabel('Position (m)')

        ax = fig.add_subplot(6, 1, 2)
        ax.plot(self.measured_states_t, self.measured_states[:, 1], 
            label=f'Aruco Y ({self.frame_frequency:.0f} Hz)', marker='x', linestyle='')
        ax.plot(self.ekf_all_states_t, self.ekf_all_states[:, 1], 
            label=f'KF Y ({self.imu_frequency:.0f} Hz)', marker='.', linestyle='', markersize=marker_size)
        ax.legend()
        ax.set_ylabel('Position (m)')

        ax = fig.add_subplot(6, 1, 3)
        ax.plot(self.measured_states_t, self.measured_states[:, 2], 
            label=f'Aruco Z ({self.frame_frequency:.0f} Hz)', marker='x', linestyle='')
        ax.plot(self.ekf_all_states_t, self.ekf_all_states[:, 2], 
            label=f'KF Z ({self.imu_frequency:.0f} Hz)', marker='.', linestyle='', markersize=marker_size)
        ax.legend()
        ax.set_ylabel('Position (m)')

        ################## Plot orientation angles ##################
        ax = plt.subplot(6, 1, 4)
        ax.plot(self.measured_states_t, rpy_meas[:, 0], 
            label=f'Aruco Roll ({self.frame_frequency:.0f} Hz)', marker='x', linestyle='')
        ax.plot(self.ekf_all_states_t, rpy_kf[:, 0], 
            label=f'KF Roll ({self.imu_frequency:.0f} Hz)', marker='.', linestyle='', markersize=marker_size)
        ax.legend()
        ax.set_ylabel('Angle (rad)')
        ax = plt.subplot(6, 1, 5)
        ax.plot(self.measured_states_t, rpy_meas[:, 1], 
            label=f'Aruco Pitch ({self.frame_frequency:.0f} Hz)', marker='x', linestyle='')
        ax.plot(self.ekf_all_states_t, rpy_kf[:, 1], 
            label=f'KF Pitch ({self.imu_frequency:.0f} Hz)', marker='.', linestyle='', markersize=marker_size)
        ax.legend()
        ax.set_ylabel('Angle (rad)')
        ax = plt.subplot(6, 1, 6)
        ax.plot(self.measured_states_t, rpy_meas[:, 2], 
            label=f'Aruco Yaw ({self.frame_frequency:.0f} Hz)', marker='x', linestyle='')
        ax.plot(self.ekf_all_states_t, rpy_kf[:, 2], 
            label=f'KF Yaw ({self.imu_frequency:.0f} Hz)', marker='.', linestyle='', markersize=marker_size)
        ax.legend()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (rad)')
        if save:
            plt.savefig(os.path.join(self.data_path, '_ekf_all.png'))
            plt.close()
        else:
            plt.show()

    def save_trajectory(self):
        raw_video_path = os.path.join(self.data_path, 'raw_video.mp4')
        raw_meta = gopro_utils.get_metadata(raw_video_path)
        pos = self.measured_states[:, :3]
        q_xyzw = self.measured_states[:, 3:7]
        rotvec = R.from_quat(q_xyzw).as_rotvec()
        pose_meas = np.concatenate([pos, rotvec], axis=-1)
        
        # Use pose interpolation to match the frame rate of the video
        pose_meas_interp = interpolation_utils.PoseInterpolator(self.measured_states_t, pose_meas)
        
        pos = self.ekf_all_states[:, :3]
        q_xyzw = self.ekf_all_states[:, 3:7]
        rotvec = R.from_quat(q_xyzw).as_rotvec()
        pose_kf = np.concatenate([pos, rotvec], axis=-1)
        pose_kf_interp = interpolation_utils.PoseInterpolator(self.ekf_all_states_t, pose_kf)

        self.n_frames = int(raw_meta['n_frames'])
        fps = float(raw_meta['fps'])
        frame_idx = np.array([i for i in range(self.n_frames)])
        timestamps = np.array([i/fps for i in range(self.n_frames)])
        poses = np.array([pose_meas_interp(t) for t in timestamps])
        pos = poses[:, :3]
        quat = R.from_rotvec(poses[:, 3:]).as_quat()

        poses_ekf = np.array([pose_kf_interp(t) for t in timestamps])
        pos_ekf = poses_ekf[:, :3]
        quat_ekf = R.from_rotvec(poses_ekf[:, 3:]).as_quat()

        p_thresh = 0.1 # meters
        q_thresh = 30.0 # degrees
        # Compute the difference in position and orientation between the ArUco and EKF data
        p_diff = np.linalg.norm(pos - pos_ekf, axis=-1)
        dot_products = np.sum(quat * quat_ekf, axis=1)
        dot_products = np.clip(dot_products, -1.0, 1.0)            
        q_diff = np.degrees(np.abs(2 * np.arccos(dot_products)))
        # Find the frames where the difference in position and orientation is greater than the threshold
        p_idx = set(np.where(p_diff > p_thresh)[0])
        q_idx = set(np.where(q_diff > q_thresh)[0])

        #is_lost = ['true' if i in p_idx or i in q_idx else 'false' for i in range(self.n_frames)]
        is_lost = ['true' if i in p_idx else 'false' for i in range(self.n_frames)]

        # Save csv file with headers: frame_idx,timestamp,state,is_lost,is_keyframe,x,y,z,q_x,q_y,q_z,q_w
        # int, float:.6f, int, bool, bool, float:.9f ...
        df = pd.DataFrame({'frame_idx': frame_idx, 
                           'timestamp': timestamps, 
                           'state': 2, # NOTE: Might need to change this
                           'is_lost': is_lost, # NOTE: Might need to change this
                           'is_keyframe': 'false', # NOTE: Might need to change this
                           'x': pos[:, 0], 'y': pos[:, 1], 'z': pos[:, 2], 
                           'q_x': quat[:, 0], 'q_y': quat[:, 1], 'q_z': quat[:, 2], 'q_w': quat[:, 3]})
        df['timestamp'] = df['timestamp'].apply(lambda x: f'{x:.6f}')
        df['x'] = df['x'].apply(lambda x: f'{x:.9f}')
        df['y'] = df['y'].apply(lambda x: f'{x:.9f}')
        df['z'] = df['z'].apply(lambda x: f'{x:.9f}')
        df['q_x'] = df['q_x'].apply(lambda x: f'{x:.9f}')
        df['q_y'] = df['q_y'].apply(lambda x: f'{x:.9f}')
        df['q_z'] = df['q_z'].apply(lambda x: f'{x:.9f}')
        df['q_w'] = df['q_w'].apply(lambda x: f'{x:.9f}')
        save_path = os.path.join(self.data_path, 'camera_trajectory.csv')
        df.to_csv(save_path, index=False)

    def make_synchronized_plot_video(self, output_path):
        '''
        Make a synchronized plot video of the ArUco cube trajectory and the estimated trajectory from the EKF
        '''
        # 3D Trajectory plot without subplot
        fig = plt.figure(figsize=(10, 8))
        ax3 = fig.add_subplot(111, projection='3d')
        ax3.set_title('3D Trajectory')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_box_aspect([1, 1, 1])

        # Set axis limits taking into account both KF and Aruco data
        min_x = min(np.min(self.ekf_all_states[:, 0]), np.min(self.measured_states[:, 0]))
        max_x = max(np.max(self.ekf_all_states[:, 0]), np.max(self.measured_states[:, 0]))
        min_y = min(np.min(self.ekf_all_states[:, 1]), np.min(self.measured_states[:, 1]))
        max_y = max(np.max(self.ekf_all_states[:, 1]), np.max(self.measured_states[:, 1]))
        min_z = min(np.min(self.ekf_all_states[:, 2]), np.min(self.measured_states[:, 2]))
        max_z = max(np.max(self.ekf_all_states[:, 2]), np.max(self.measured_states[:, 2]))

        max_range = np.array([max_x - min_x, max_y - min_y, max_z - min_z]).max() / 2.0
        mid_x = (max_x + min_x) * 0.5
        mid_y = (max_y + min_y) * 0.5
        mid_z = (max_z + min_z) * 0.5
        ax3.set_xlim(mid_x - max_range, mid_x + max_range)
        ax3.set_ylim(mid_y - max_range, mid_y + max_range)
        ax3.set_zlim(mid_z - max_range, mid_z + max_range)

        # Draw a frame at the origin
        origin = np.eye(4) * 0.1
        ax3.quiver(origin[0, 3], origin[1, 3], origin[2, 3], origin[0, 0], origin[1, 0], origin[2, 0], color='r')
        ax3.quiver(origin[0, 3], origin[1, 3], origin[2, 3], origin[0, 1], origin[1, 1], origin[2, 1], color='g')
        ax3.quiver(origin[0, 3], origin[1, 3], origin[2, 3], origin[0, 2], origin[1, 2], origin[2, 2], color='b')

        # Prepare video writer
        width, height = fig.get_size_inches() * fig.dpi
        video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.frame_frequency, (int(width), int(height)))

        # Create buffer outside the loop to reuse it
        buf = BytesIO()

        aruco_line, = ax3.plot([], [], [], label=f'Aruco Trajectory ({self.frame_frequency:.0f} Hz)', marker='x', linestyle='')
        ekf_line, = ax3.plot([], [], [], label=f'KF Trajectory ({self.imu_frequency:.0f} Hz)', marker='.', linestyle='', markersize=1)
        ax3.legend()
        for i in range(int(self.n_frames)):
            timestep = i / self.frame_frequency
            ekf_index = np.argmin(np.abs(self.ekf_all_states_t - timestep))
            measured_index = np.argmin(np.abs(self.measured_states_t - timestep))
            
            # Update plot data
            aruco_line.set_data(self.measured_states[:measured_index, 0], self.measured_states[:measured_index, 1])
            aruco_line.set_3d_properties(self.measured_states[:measured_index, 2])
            
            ekf_line.set_data(self.ekf_all_states[:ekf_index, 0], self.ekf_all_states[:ekf_index, 1])
            ekf_line.set_3d_properties(self.ekf_all_states[:ekf_index, 2])

            # Save the plot to an in-memory bytes buffer
            buf.seek(0)
            buf.truncate()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Read the image from the buffer and convert it to a NumPy array
            img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            video.write(img)

        plt.close(fig)
        cv2.destroyAllWindows()
        video.release()
        buf.close()

        print(f'Video saved successfully as {output_path}:\
              \nTotal frames | video duration: {int(self.n_frames)} frames | {self.n_frames / self.frame_frequency} s\
              \nResolution: {int(width)} x {int(height)} pixels\
              \nFrame rate: {self.frame_frequency:.0f} Hz')

@click.command()
@click.option('-i', '--input_path', required=True, help='Path to the data directory')
@click.option('-m', '--mode', type=str, required=True, help='pruner or gripper')
@click.option('-v', '--visualize', is_flag=True, default=True, help='Visualize the imported data')
@click.option('-o', '--output_path', default=None, help='Output path for the synchronized plot video')
def main(input_path, mode, visualize, output_path):

    # Initialization parameters
    sigma_accel = 0.224 #0.124  # [m/s^2] (value derived from Noise Spectral Density in datasheet)
    sigma_gyro = 0.0276 #0.00276  # [rad/s] (value derived from Noise Spectral Density in datasheet)
    sigma_init_pos = 1.0  # [m]
    sigma_init_vel = 0.1  # [m/s]
    sigma_init_dtheta = 1.0  # [rad]

    eskf = ESKF(sigma_accel, sigma_gyro, sigma_init_pos, sigma_init_vel, sigma_init_dtheta, mode)
    eskf.import_data(input_path, visualize=visualize)
    eskf.run()    
    eskf.convert_frame_from_imu_to_camera()
    eskf.save_trajectory()
    eskf.plot_results(save=visualize)
    if output_path:
        eskf.make_synchronized_plot_video(output_path)

if __name__=='__main__':
    main()
