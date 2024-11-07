#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Required Environment to run this example :
#    - Protocol 2.0 supported DYNAMIXEL(X, P, PRO/PRO(A), MX 2.0 series)
#    - DYNAMIXEL Starter Set (U2D2, U2D2 PHB, 12V SMPS)
#  How to use the example :
#    - Select the DYNAMIXEL in use at the MY_DXL in the example code. 
#    - Build and Run from proper architecture subdirectory.
#    - For ARM based SBCs such as Raspberry Pi, use linux_sbc subdirectory to build and run.
#    - https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/overview/

import dynamixel_sdk as dynamixel
import time

class GripperDriver(object):
    def __init__(self):
        # Control table address and byte length
        self.ADDR_OPERATING_MODE, self.LEN_OPERATING_MODE         = 11, 1 
        self.ADDR_CURRENT_LIMIT, self.LEN_CURRENT_LIMIT           = 38, 2
        self.ADDR_TORQUE_ENABLE, self.LEN_TORQUE_ENABLE           = 64, 1
        self.ADDR_GOAL_PWM, self.LEN_GOAL_PWM                     = 100, 2
        self.ADDR_GOAL_CURRENT, self.LEN_GOAL_CURRENT             = 102, 2
        self.ADDR_PROFILE_VELOCITY, self.LEN_PROFILE_VELOCITY     = 112, 4
        self.ADDR_GOAL_POSITION, self.LEN_GOAL_POSITION           = 116, 4
        self.ADDR_MOVING, self.LEN_MOVING                         = 122, 1
        self.ADDR_PRESENT_PWM, self.LEN_PRESENT_PWM               = 124, 2
        self.ADDR_PRESENT_CURRENT, self.LEN_PRESENT_CURRENT       = 126, 2
        self.ADDR_PRESENT_VELOCITY, self.LEN_PRESENT_VELOCITY     = 128, 4
        self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION     = 132, 4

        # Operating Modes
        # https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/#operating-mode
        self.OPERATING_MODE = {
            'current': 0,
            'velocity': 1,
            'position': 3, # Default
            'extended_position': 4,
            'current_based_position': 5,
            'pwm': 16
        }

        # Use the actual port assigned to the U2D2.
        # ex) Windows: "COM*", Linux: "/dev/ttyUSB*", Mac: "/dev/tty.usbserial-*"
        self.DEVICENAME                  = '/dev/ttyUSB0'
        self.PROTOCOL_VERSION            = 2.0 # https://emanual.robotis.com/docs/en/dxl/protocol2/
        self.BAUDRATE                    = 57600
        self.DXL_ID                      = 1 # Factory default ID of all DYNAMIXEL is 1

        # Unit conversion
        self.current_per_unit = 3.36 # 3.36mA per unit
        self.pwm_per_unit = 0.113 # 0.113% per unit
        self.velocity_per_unit = 0.229 # 0.229rpm per unit
        self.position_per_unit = 360.0/4096 # 0.088 degrees per unit 

        self.pos_threshold = 20
        self.calibrated = False
        self.init_port()

    ############################ Initialization ############################
    def init_port(self):
        self.port_handler = dynamixel.PortHandler(self.DEVICENAME)
        self.packet_handler = dynamixel.PacketHandler(self.PROTOCOL_VERSION)

        if self.port_handler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            quit()
    
        if self.port_handler.setBaudRate(self.BAUDRATE):
            print(f"Succeeded to change the baudrate: {self.BAUDRATE}")
        else:
            print("Failed to change the baudrate")
            quit()
        time.sleep(0.1)

    def close(self):
        self.port_handler.closePort()
        print("Gripper control closed")

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    ############################ Utility ############################
    def _check_results(self, dxl_comm_result, dxl_error, address):
        if dxl_comm_result != dynamixel.COMM_SUCCESS:
            print(f"Error Address {address}: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"Error Address {address}: {self.packet_handler.getRxPacketError(dxl_error)}")
        else:
            return True # No error
        return False # Error

    def _item_write(self, address, data, length):
        '''
        Writes data to the specified register for a given motor
        id - Dynamixel ID to write data to
        address - register number
        data - data to write
        length - size of register in bytes
        return <bool> - true if data was successfully written; false otherwise
        '''
        if length == 1:
            dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, self.DXL_ID, address, data)
        elif length == 2:
            dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(self.port_handler, self.DXL_ID, address, data)
        elif length == 4:
            dxl_comm_result, dxl_error = self.packet_handler.write4ByteTxRx(self.port_handler, self.DXL_ID, address, data)
        else:
            print("Invalid data length...")
            return False
        return self._check_results(dxl_comm_result, dxl_error, address)

    def _item_read(self, address, length):
        '''
        Reads data from the specified register for a given motor
        id - Dynamixel ID to read data from
        address - register number
        length - size of register in bytes
        return state - variable to store the requested data
        return <bool> - true if data was successfully retrieved; false otherwise
        NOTE: DynamixelSDK uses 2's complement so we need to check to see if 'state' should be negative (hex numbers)
        '''
        if length == 1:
            state, dxl_comm_result, dxl_error = self.packet_handler.read1ByteTxRx(self.port_handler, self.DXL_ID, address)
        elif length == 2:
            state, dxl_comm_result, dxl_error = self.packet_handler.read2ByteTxRx(self.port_handler, self.DXL_ID, address)
            if state > 0x7fff:
                state = state - 65536
        elif length == 4:
            state, dxl_comm_result, dxl_error = self.packet_handler.read4ByteTxRx(self.port_handler, self.DXL_ID, address)
            if state > 0x7fffffff:
                state = state - 4294967296
        else:
            print("Invalid data length...")
            return 0, False
        return state, self._check_results(dxl_comm_result, dxl_error, address)        
    
    ############################ Setters ############################
    def set_position(self, position, wait=False):
        res = self._item_write(self.ADDR_GOAL_POSITION, int(position), self.LEN_GOAL_POSITION)
        moving = wait
        while moving:
            time.sleep(0.1)
            moving, _ = self.get_moving()
        return res
    
    def set_velocity(self, velocity):
        # Valid input range: -234rpm - 234rpm
        velocity = min(max(velocity, -234), 234)
        velocity = int(velocity / self.velocity_per_unit)
        return self._item_write(self.ADDR_PROFILE_VELOCITY, velocity, self.LEN_PROFILE_VELOCITY)
    
    def set_current(self, torque):
        # Valid input range: -6500mA - 6500mA
        torque = min(max(torque, -6500), 6500)
        torque = int(torque / self.current_per_unit)
        return self._item_write(self.ADDR_GOAL_CURRENT, torque, self.LEN_GOAL_CURRENT)

    def set_pwm(self, pwm):
        # Valid input range: -100%-100%
        pwm = min(max(pwm, -100), 100)
        pwm = int(pwm / self.pwm_per_unit)
        return self._item_write(self.ADDR_GOAL_PWM, pwm, self.LEN_GOAL_PWM)
    
    def set_operating_mode(self, mode):
        res = self.set_torque_enable(0)
        res = self._item_write(self.ADDR_OPERATING_MODE, mode, self.LEN_OPERATING_MODE) and res
        res = self.set_torque_enable(1) and res
        return res
    
    def set_torque_enable(self, enable):
        return self._item_write(self.ADDR_TORQUE_ENABLE, enable, self.LEN_TORQUE_ENABLE)
    
    ############################ Getters ############################
    def get_position(self):
        position, res = self._item_read(self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)
        for i in range(10):
            if res:
                break
            position, res = self._item_read(self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)
        if not res:
            self.close()
            print("Failed to get current position. Closing gripper control.")
        return position, res
    
    def get_velocity(self):
        # Returns velocity in rpm
        velocity, res = self._item_read(self.ADDR_PRESENT_VELOCITY, self.LEN_PRESENT_VELOCITY)
        velocity = velocity * self.velocity_per_unit
        return velocity, res
    
    def get_current(self):
        # Returns current in mA
        current, res = self._item_read(self.ADDR_PRESENT_CURRENT, self.LEN_PRESENT_CURRENT)
        current = current * self.current_per_unit
        return current, res
    
    def get_pwm(self):
        # Returns pwm in %
        pwm, res = self._item_read(self.ADDR_PRESENT_PWM, self.LEN_PRESENT_PWM)
        pwm = pwm * self.pwm_per_unit
        return pwm, res
    
    def get_current_limit(self):
        # Returns current limit in mA
        limit, res = self._item_read(self.ADDR_CURRENT_LIMIT, self.LEN_CURRENT_LIMIT)
        limit = limit * self.current_per_unit
        return limit, res

    def get_operating_mode(self):
        mode, res = self._item_read(self.ADDR_OPERATING_MODE, self.LEN_OPERATING_MODE)
        return mode, res
    
    def get_moving(self):
        moving, res = self._item_read(self.ADDR_MOVING, self.LEN_MOVING)
        return moving, res
    
    ############################ Control ############################
    def calibrate_gripper(self):
        print("Calibrating gripper...", end=" ")
        self.set_operating_mode(self.OPERATING_MODE['current'])
        self.set_current(50) # NOTE: Hardcoded value
        time.sleep(1.5) # NOTE: Hardcoded value
        position, res = self.get_position()
        self.closed_pos = position
        self.open_pos = self.closed_pos - 1224 # NOTE: Hardcoded value
        self.calibrated = True
        self.set_operating_mode(self.OPERATING_MODE['extended_position'])
        self.open_gripper(wait=True)
        print(f'Finished! Open position: {self.open_pos}, Closed position: {self.closed_pos}')

    def open_gripper(self, wait=False):
        assert self.calibrated, "Gripper not calibrated!"
        res = self.set_position(self.open_pos, wait)
        return res
    
    def close_gripper(self, wait=False):
        assert self.calibrated, "Gripper not calibrated!"
        res = self.set_position(self.closed_pos, wait)
        return res
    
    def get_state(self):
        while True:
            try:
                position, res_p = self.get_position() # Position in encoder units
                # Determine the position of the gripper wrt the open and closed positions as a fraction
                position = (position - self.open_pos) / (self.closed_pos - self.open_pos) 
                position = 0.0 if position < 0.1 else 1.0
                #velocity, res_v = self.get_velocity() # Velocity in rpm
                #current, res_c = self.get_current() # Current in mA
                res = res_p #and res_v and res_c
                assert res
                state = {
                    'position': position,
                    'velocity': 0, #velocity,
                    'force': 0, #current, 
                }

                break
            except:
                pass
        return state
    
    def set_gripper_width(self, width, wait=False):
        assert self.calibrated, "Gripper not calibrated!"
        width = min(max(width, 0.0), 1.0)
        width = 0.0 if width < 0.1 else 1.0
        position = self.open_pos + width * (self.closed_pos - self.open_pos)
        res = self.set_position(position, wait)
        return self.get_state()
    
def main():
    with GripperDriver() as gripper:
        try:
            gripper.calibrate_gripper()
            #gripper.set_position(0, wait=True)
            #print(gripper.get_position())
            assert gripper.close_gripper(wait=True)
            assert gripper.open_gripper(wait=True)
            #assert gripper.set_gripper_width(width=0.5, wait=True)
            #assert gripper.set_gripper_width(width=0.0, wait=True)
            #assert gripper.set_gripper_width(width=0.9, wait=True)
            #assert gripper.open_gripper(wait=True)
        except:
            gripper.close()
            quit()

if __name__ == '__main__':
    main()