# -*- coding: utf-8 -*-
import math
import numpy as np


def rpy2rot_xyz(roll, pitch, yaw):
    # vrep按照xyz顺序内旋，依次右乘，因此是xyz
    R_x = np.array([[1,                            0,               0],
                    [0,               math.cos(roll), -math.sin(roll)],
                    [0,               math.sin(roll),  math.cos(roll)]])              
    R_y = np.array([[ math.cos(pitch),             0,  math.sin(pitch)],
                    [               0,             1,                0],
                    [-math.sin(pitch),             0,  math.cos(pitch)]])        
    R_z = np.array([[math.cos(yaw),   -math.sin(yaw),               0],
                    [math.sin(yaw),    math.cos(yaw),               0],
                    [            0,                0,               1]])      
    rot = np.dot(R_x, np.dot(R_y, R_z))
    return rot

def rpy2rot_zyx(roll, pitch, yaw):
    # UR5实物按zyx内旋，依次右乘，因此是zyx
    R_x = np.array([[1,                            0,               0],
                    [0,               math.cos(roll), -math.sin(roll)],
                    [0,               math.sin(roll),  math.cos(roll)]])              
    R_y = np.array([[ math.cos(pitch),             0,  math.sin(pitch)],
                    [               0,             1,                0],
                    [-math.sin(pitch),             0,  math.cos(pitch)]])        
    R_z = np.array([[math.cos(yaw),   -math.sin(yaw),               0],
                    [math.sin(yaw),    math.cos(yaw),               0],
                    [            0,                0,               1]])      
    rot = np.dot(R_z, np.dot(R_y, R_x))
    return rot


def rot2euler(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
 
    singular = sy < 1e-6
 
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
 
    return np.array([x, y, z])


def posort2homogeneous(pos, rot):
    "位置、姿态转为4*4齐次变换矩阵"
    # 创建齐次变换矩阵
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    
    return T


def homogeneous2posort(homogeneous):
    "4*4齐次变换矩阵转为6维位姿向量"
    rot = homogeneous[:3, :3]
    pos = homogeneous[:3, 3]
    
    rpy = rot2euler(rot)
    pos_ort = np.concatenate([pos, rpy], axis=0)

    return pos_ort
    

def tcp2base(tcp_pos, delta_pos, sim=False, real=True):
    "将六维位姿变化量从TCP坐标系转换到基坐标系下"
    if sim:
        tcp_rot = rpy2rot_xyz(tcp_pos[3], tcp_pos[4], tcp_pos[5])
        delta_rot = rpy2rot_xyz(delta_pos[3], delta_pos[4], delta_pos[5])
    elif real:
        tcp_rot = rpy2rot_zyx(tcp_pos[3], tcp_pos[4], tcp_pos[5])
        delta_rot = rpy2rot_zyx(delta_pos[3], delta_pos[4], delta_pos[5])
    
    # TCP坐标系在基坐标系下的齐次变换矩阵
    T_TCP_base = posort2homogeneous(tcp_pos[0:3], tcp_rot)
    # T_base_TCP = np.linalg.inv(T_TCP_base)
    # 六维位姿变化量在TCP坐标系下的齐次变换矩阵
    T_deltapos_TCP = posort2homogeneous(delta_pos[0:3], delta_rot)
    # 六维位姿变化量在基坐标系下的齐次变换矩阵
    T_deltapos_base = np.dot(T_TCP_base, T_deltapos_TCP)
    deltapos_base = homogeneous2posort(T_deltapos_base)

    return deltapos_base


def control_rad(orientation):
    for i in range(len(orientation)):
        if orientation[i] < -1*math.pi:
            orientation[i] = orientation[i]+2*math.pi
        elif orientation[i] > math.pi:
            orientation[i] = orientation[i]-2*math.pi
        elif 3.13 <= abs(orientation[i]) <= math.pi:
            orientation[i] = abs(orientation[i])
    
    return orientation


if __name__ == "__main__":
    (roll, pitch ,yaw) = (3.14, 0 -0)
    print(rpy2rot_xyz(roll, pitch ,yaw))

