# encoding: utf-8
from support_files import sim
import time
import math
import numpy as np
import cv2
from robot_control import utils

class UR5_Sim(object):
    def __init__(self):
        self.RAD2DEG = 180/math.pi

        self.jointNum = 6
        self.baseName = 'UR5'
        self.rgName = 'RG2_openCloseJoint'
        self.jointName = "UR5_joint"
        self.ftsensorName = 'FT_sensor'
        self.targetName = 'UR5_target'
        self.tipName = 'UR5_tip'
        self.graspName = 'RG2_attachPoint'


        # 远程连接coppeliasim
        print("Simulation started")
        sim.simxFinish(-1)     # 关闭潜在的连接
        # 每隔0.2s检测一次, 直到连接上V-rep
        while True:
            # simxStart的参数分别为：服务端IP地址(连接本机用127.0.0.1);端口号;是否等待服务端开启;连接丢失时是否尝试再次连接;超时时间(ms);数据传输间隔(越小越快)
            self.clientID = sim.simxStart('127.0.0.1', 19995, True, True, 5000, 5)
            if self.clientID > -1:
                print("Connection success!")
                break
            else:
                time.sleep(0.2)
                print("Failed connecting to remote API server!")
                print("Maybe you forget to run the simulation on vrep...")

        sim.simxStartSimulation(self.clientID, sim.simx_opmode_oneshot)    # 仿真初始化
        
        self.jointHandle = np.zeros((self.jointNum, 1), dtype=int)
        # print("jointHandle = ", self.jointHandle)
        for i in range(self.jointNum):
            _, returnHandle = sim.simxGetObjectHandle(self.clientID, self.jointName + str(i+1), sim.simx_opmode_blocking)
            self.jointHandle[i] = returnHandle

        _, self.baseHandle = sim.simxGetObjectHandle(self.clientID, self.baseName, sim.simx_opmode_blocking)
        _, self.rgHandle = sim.simxGetObjectHandle(self.clientID, self.rgName, sim.simx_opmode_blocking)
        _, self.ftsensorHandle = sim.simxGetObjectHandle(self.clientID, self.ftsensorName, sim.simx_opmode_blocking)
        _, self.targetHandle = sim.simxGetObjectHandle(self.clientID, self.targetName, sim.simx_opmode_blocking)
        _, self.tipHandle = sim.simxGetObjectHandle(self.clientID, self.tipName, sim.simx_opmode_blocking)
        _, self.graspHandle = sim.simxGetObjectHandle(self.clientID, self.graspName, sim.simx_opmode_blocking)


    def __del__(self):
        sim.simxFinish(self.clientID)
        print("Simulation End!")

    
    def getJointsAngle(self):
        joints_position = []
        for i in range(self.jointNum):
            _, jpos = sim.simxGetJointPosition(self.clientID, self.jointHandle[i], sim.simx_opmode_blocking)
            joints_position.append(jpos*self.RAD2DEG)
        
        return joints_position
    

    def getPosOrt(self, objectName, relativeName=None):
        _, objectHandle = sim.simxGetObjectHandle(self.clientID, objectName, sim.simx_opmode_blocking)

        if relativeName is None:
            relativeHandle = self.baseHandle
        else:
            _, relativeHandle = sim.simxGetObjectHandle(self.clientID, relativeName, sim.simx_opmode_blocking)

        _, position = sim.simxGetObjectPosition(self.clientID, objectHandle, relativeHandle, sim.simx_opmode_blocking)
        _, orientation = sim.simxGetObjectOrientation(self.clientID, objectHandle, relativeHandle, sim.simx_opmode_blocking)
        
        return position, orientation
    
    
    def getGraspPosOrt(self, objectName):
        _, objectHandle = sim.simxGetObjectHandle(self.clientID, objectName, sim.simx_opmode_blocking)

        _, position = sim.simxGetObjectPosition(self.clientID, objectHandle, self.graspHandle, sim.simx_opmode_blocking)
        _, orientation = sim.simxGetObjectOrientation(self.clientID, objectHandle, self.graspHandle, sim.simx_opmode_blocking)
        
        return position, orientation
    

    def getVisionImage(self, cameraName='Camera_y', clipArr=[75, 160, 70, 190]):
        _, self.cameraHandle = sim.simxGetObjectHandle(self.clientID, cameraName, sim.simx_opmode_blocking)
        _, resolution, rawImage = sim.simxGetVisionSensorImage(self.clientID, self.cameraHandle, 0, sim.simx_opmode_blocking)
        color_img = np.asarray(rawImage)
        # 重塑img数组形状:(高度, 宽度, 颜色通道(r、g、b))
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float)
        # 对vrep中返回的数据格式中负值的颜色编码进行修正
        color_img[color_img < 0] += 255
        color_img = np.flipud(color_img)
        color_img = color_img.astype(np.uint8)

        color_img = color_img[clipArr[0]:clipArr[1], clipArr[2]:clipArr[3]]

        # cv2.imshow('CoppeliaSim Image', color_img)
        # cv2.waitKey(0)

        return color_img

    
    def setObjectPosOrt(self, objectName, position, orientation):
        _, objectHandle = sim.simxGetObjectHandle(self.clientID, objectName, sim.simx_opmode_blocking)
        sim.simxSetObjectPosition(self.clientID, objectHandle, self.baseHandle, position, sim.simx_opmode_blocking)
        sim.simxSetObjectOrientation(self.clientID, objectHandle, self.baseHandle, orientation, sim.simx_opmode_blocking)

    def setGraspPosOrt(self, objectName, position, orientation):
        _, objectHandle = sim.simxGetObjectHandle(self.clientID, objectName, sim.simx_opmode_blocking)
        sim.simxSetObjectPosition(self.clientID, objectHandle, self.graspHandle, position, sim.simx_opmode_blocking)
        sim.simxSetObjectOrientation(self.clientID, objectHandle, self.graspHandle, orientation, sim.simx_opmode_blocking)


    def moveFK(self, joints_position):
        for i in range(self.jointNum):
            sim.simxSetJointTargetPosition(self.clientID, self.jointHandle[i], joints_position[i], sim.simx_opmode_blocking)


    def moveIK(self, position, orientation, isWait=True, isSoft=False, f_th=15):
        sim.simxSetObjectPosition(self.clientID, self.targetHandle, self.baseHandle, position, sim.simx_opmode_blocking)
        sim.simxSetObjectOrientation(self.clientID, self.targetHandle, self.baseHandle, orientation, sim.simx_opmode_blocking)
        # print('target', position, orientation)

        cnt = 0
        search_flag = False
        soft_flag = False
        # 直到完全移动到位，再停止运行此函数
        while isWait:
            if isSoft:
                ft = self.gravityCompensate()
                # if ft[2] > -0.5:
                #     search_flag = True
                #     break
                for i in range(3):
                    if abs(ft[i]) > f_th or abs(ft[i+3]) > f_th/3:
                        soft_flag = True
                        position[2] += 1e-4
                        self.moveIK(position, orientation)
                        # print('-----MoveIK soft!')
                        # print(f'moveIK target = {position}, {orientation}')
                        break
                if soft_flag:
                    break 

            cnt += 1
            cur_pos, cur_orient = self.getPosOrt(self.tipName)
            # print('current', cur_pos, cur_orient)
            #     continue
            diff1 = sum(abs(x-y) for x,y in zip(position, cur_pos))
            # 修正-3.14153 和 3.14152 错误判断的情况
            diff2 = sum(min(abs(x-y), abs(x-y-2*math.pi), abs(x-y+2*math.pi)) for x,y in zip(orientation, cur_orient))
            if (diff1  < 2e-4 and diff2 < 3e-4) or cnt > 100:
                break
        
        # cur_pos, cur_orient = self.getPosOrt(self.tipName)
        # pos_err = [cur_pos[i]-position[i] for i in range(3)]
        # print('--------------------------')
        # print(f'pos_err = {pos_err}')
        # print(f'soft_flag = {soft_flag}, cnt = {cnt}\n')

        return soft_flag, search_flag
    

    def moveSelfIK(self, position, orientation, isWait=True, isSoft=False, f_th=15):
        "相对于TCP坐标系进行运动，发送的位姿直接为位姿增量"
        sim.simxSetObjectPosition(self.clientID, self.targetHandle, self.targetHandle, position, sim.simx_opmode_blocking)
        sim.simxSetObjectOrientation(self.clientID, self.targetHandle, self.targetHandle, orientation, sim.simx_opmode_blocking)
        # print('target', position, orientation)

        cnt = 0
        search_flag = False
        soft_flag = False
        # 直到完全移动到位，再停止运行此函数
        while isWait:
            if isSoft:
                ft = self.gravityCompensate()
                # if ft[2] > -0.5:
                #     search_flag = True
                #     break
                for i in range(3):
                    if abs(ft[i]) > f_th or abs(ft[i+3]) > f_th/3:
                        soft_flag = True
                        # position[2] += 1e-4
                        # self.moveIK(position, orientation)
                        # print('-----MoveIK soft!')
                        # print(f'moveIK target = {position}, {orientation}')
                        break
                if soft_flag:
                    break 

            cnt += 1
            cur_pos, cur_orient = self.getPosOrt(self.tipName, 'UR5_target')
            # print('current', cur_pos, cur_orient)
            #     continue
            diff1 = np.linalg.norm(np.array(cur_pos), ord=1)
            # 修正-3.14153 和 3.14152 错误判断的情况
            diff2 = sum(min(abs(x), abs(x-2*math.pi), abs(x+2*math.pi)) for x in cur_orient)
            if (diff1  < 2e-4 and diff2 < 3e-4) or cnt > 100:
                break
        
        # cur_pos, cur_orient = self.getPosOrt(self.tipName)
        # pos_err = [cur_pos[i]-position[i] for i in range(3)]
        # print('--------------------------')
        # print(f'pos_err = {pos_err}')
        # print(f'soft_flag = {soft_flag}, cnt = {cnt}\n')

        return soft_flag, search_flag
    

    def getFTData(self):
        for _ in range(2):
            _, _, forceRaw, torqueRaw = sim.simxReadForceSensor(self.clientID, self.ftsensorHandle, sim.simx_opmode_streaming)

        return forceRaw, torqueRaw
    

    def gravityIdentifyParam(self, objectName='peg'):
        "力矩传感器的重力辨识参数"
        param_dicts = {
            'peg':[6.67, -1.41431990e-04, -6.57575598e-05, 1.55953757e-01],
            'circlepeg':[6.67, -1.36713353e-04, -5.19256416e-06, 1.58539373e-01],
            'sixpeg':[6.67, -1.23013503e-04, 1.14656750e-05, 1.56662654e-01],
            'dualpeg':[6.67, -1.40419176e-04, 4.86886150e-06, 1.54300125e-01]
        }

        Gt = param_dicts[objectName][0]

        self.o_x = param_dicts[objectName][1]
        self.o_y = param_dicts[objectName][2]
        self.o_z = param_dicts[objectName][3]

        self.F_Gb = np.array([[0],[0],[-Gt]])    # 工具重力在基坐标系


    def gravityCompensate(self, objectName='circlepeg', isPrint=False):
        "力矩传感器重力补偿"
        # 解决simxReadForceSensor、simxGetObjectOrientation首次调用时读数为0的问题
        for _ in range(2):
            _, _, forceRaw, torqueRaw = sim.simxReadForceSensor(self.clientID, self.ftsensorHandle, sim.simx_opmode_streaming)
            # time.sleep(0.001)
            # print("ftsensor = ", forceRaw, torqueRaw)
            _, ftsensor_orientation = sim.simxGetObjectOrientation(self.clientID, self.ftsensorHandle, self.baseHandle, sim.simx_opmode_blocking)
        x = ftsensor_orientation[0]
        y = ftsensor_orientation[1]
        z = ftsensor_orientation[2]
        # print(x, y, z)

        # 获取夹取该物体的辨识参数
        self.gravityIdentifyParam(objectName)

        # 计算工具重力的分量
        R_sb = utils.rpy2rot_xyz(x, y, z)     # 基坐标系和传感器坐标系的旋转矩阵
        F_Gs = np.dot(np.linalg.inv(R_sb), self.F_Gb)
        o_xyz = np.array([[        0, -self.o_z,  self.o_y],
                          [ self.o_z,         0, -self.o_x],
                          [-self.o_y,  self.o_x,         0]])
        M_Gs = np.dot(o_xyz, F_Gs)

        # 计算标定后的力矩数据
        fx_calib = forceRaw[0] - F_Gs[0][0]
        fy_calib = forceRaw[1] - F_Gs[1][0]
        fz_calib = forceRaw[2] - F_Gs[2][0]
        tx_calib = torqueRaw[0] - M_Gs[0][0]
        ty_calib = torqueRaw[1] - M_Gs[1][0]
        tz_calib = torqueRaw[2] - M_Gs[2][0]

        ft = np.array([fx_calib,fy_calib,fz_calib,tx_calib,ty_calib,tz_calib])

        if isPrint is True:
            print("\n----------力矩传感器重力补偿----------")
            print("未补偿:\n力:x=%f,y=%f,z=%f\n力矩:x=%f,y=%f,z=%f"
                    %(forceRaw[0],forceRaw[1],forceRaw[2],torqueRaw[0],torqueRaw[1],torqueRaw[2]))
            print("补偿后:\n力:x=%f,y=%f,z=%f\n力矩:x=%f,y=%f,z=%f"
                    %(fx_calib,fy_calib,fz_calib,tx_calib,ty_calib,tz_calib))
            print("---------------------------------------\n")
            
        return ft
    

    def closeRG2(self, gripper_motor_velocity=-0.1, gripper_motor_force=5):
        _, gripper_joint_position = sim.simxGetJointPosition(self.clientID, self.rgHandle, sim.simx_opmode_blocking)

        sim.simxSetJointForce(self.clientID, self.rgHandle, gripper_motor_force, sim.simx_opmode_blocking)
        sim.simxSetJointTargetVelocity(self.clientID, self.rgHandle, gripper_motor_velocity, sim.simx_opmode_blocking)
        gripper_fully_closed = False
        while gripper_joint_position > -0.045:  # Block until gripper is fully closed
            _, new_gripper_joint_position = sim.simxGetJointPosition(self.clientID, self.rgHandle, sim.simx_opmode_blocking)
            # print('\n----------------------------------')
            # print(f'gripper_joint_position = {gripper_joint_position}\nnew_gripper_joint_position = {new_gripper_joint_position}')
            if new_gripper_joint_position >= gripper_joint_position:
                return gripper_fully_closed
            gripper_joint_position = new_gripper_joint_position
        
        gripper_fully_closed = True
        return gripper_fully_closed


    def openRG2(self, gripper_motor_velocity=0.1, gripper_motor_force=5):
        _, gripper_joint_position = sim.simxGetJointPosition(self.clientID, self.rgHandle, sim.simx_opmode_blocking)
        sim.simxSetJointForce(self.clientID, self.rgHandle, gripper_motor_force, sim.simx_opmode_blocking)
        sim.simxSetJointTargetVelocity(self.clientID, self.rgHandle, gripper_motor_velocity, sim.simx_opmode_blocking)
        while gripper_joint_position < 0.01:  # Block until gripper is fully open
            _, gripper_joint_position = sim.simxGetJointPosition(self.clientID, self.rgHandle, sim.simx_opmode_blocking)


    def setPropertyDynamic(self, objectName):
        # 只能设置为不动， 不能恢复为动态
        _, objectHandle = sim.simxGetObjectHandle(self.clientID, objectName, sim.simx_opmode_blocking)
        sim.simxSetModelProperty(self.clientID, objectHandle, sim.sim_modelproperty_not_dynamic, sim.simx_opmode_blocking)


    def setPropertyRespondable(self, objectName):
        # 只能设置为不响应，不能恢复为响应
        _, objectHandle = sim.simxGetObjectHandle(self.clientID, objectName, sim.simx_opmode_blocking)
        sim.simxSetModelProperty(self.clientID, objectHandle, sim.sim_modelproperty_not_collidable, sim.simx_opmode_blocking)


    def setObjectParent(self, objectName, parentName):
        _, objectHandle = sim.simxGetObjectHandle(self.clientID, objectName, sim.simx_opmode_blocking)
        if parentName == -1:
            sim.simxSetObjectParent(self.clientID, objectHandle, -1, False, sim.simx_opmode_blocking)
        else:
            _, parentHandle = sim.simxGetObjectHandle(self.clientID, parentName, sim.simx_opmode_blocking)
            sim.simxSetObjectParent(self.clientID, objectHandle, parentHandle, True, sim.simx_opmode_blocking)


    def open(self):
        sim.simxSetIntegerSignal(self.clientID, 'RG2_open', 1, sim.simx_opmode_blocking)


if __name__ == '__main__':
    robot = UR5_Sim()
    # print("position, orientation = ", robot.getPosOrient('UR5_tip'))
    # print("joints = ", robot.getJointsAngle())
    # robot.openRG2()
    # time.sleep(1)
    robot.closeRG2()
    # robot.setObjectParent('peg0', 'RG2_attachPoint')
    # robot.setProperty('lock_base')
    # robot.gravityCompensate()
    