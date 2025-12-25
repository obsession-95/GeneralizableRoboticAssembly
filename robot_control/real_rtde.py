# encoding: utf-8
import rtde_control
import rtde_receive
import urx
import numpy as np
import cv2
import math
from robot_control import utils, ftsensor_read, camera_read
from collections import deque
import time


class UR5_Rtde(object):
    def __init__(self, isRealsense=True, isZed=True):
        UR5_IP = "192.168.1.21"
        self.rtde_c = rtde_control.RTDEControlInterface(UR5_IP)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(UR5_IP)
        self.rob = urx.Robot(UR5_IP)
        self.camera = camera_read.CameraReader(isRealsense=isRealsense, isZed=isZed)
        self.ft_buffer = None
        


    def close(self):
        self.rtde_c.stopScript()
        self.rob.close()
        print("Robot Connection End")
        self.camera.close()
        print("Camera Connection End")

    def refresh(self):
        self.rtde_c.disconnect()
        self.rtde_c.reconnect()


    def getPosRotvec(self):
        pos_rotvec = self.rtde_r.getActualTCPPose()
        # 获取位置和旋转向量
        return pos_rotvec    
        
    def getPosOrt(self):
        pos_rotvec = self.rtde_r.getActualTCPPose()
        rotvector = np.array(pos_rotvec[3:6]).reshape(3, 1)
        rot = cv2.Rodrigues(rotvector)[0]
        rpy = utils.rot2euler(rot).tolist()

        pos_ort = np.array(pos_rotvec[0:3] + rpy)
        # 获取位置和rpy角
        return pos_ort


    def setTCPOffset(self, tcp_offset):
        # 设置TCP
        self.rtde_c.setTcp(tcp_offset)

    def getTCPOffset(self):
        return self.rtde_c.getTCPOffset()

    def setPayload(self, mass, cog):
        # mass: Mass in kilograms
        # cog: Center of Gravity, a vector [CoGx, CoGy, CoGz] 
        #      specifying the displacement (in meters) from the toolmount.
        self.rtde_c.setTargetPayload(mass=mass, cog=cog)


    def setPosRotvec(self, pos_rotvec, speed=0.5, acc=0.25, asynchronous=True, 
                        isWait=True, isSoft=False):
        if asynchronous is False:
            # 不是异步通信模式，无法在执行中中断
            isWait = False
            isSoft = False          
        self.rtde_c.moveL(pos_rotvec, speed, acc, asynchronous)
        start_time = time.time()

        cnt = 0
        soft_flag = False
        # 直到完全移动到位，或力/力矩大于阈值，再停止运行此函数
        while isWait:
            if isSoft:
                ft =self.getFTdata()
                for i in range(3):
                    if abs(ft[i]) > 15 or abs(ft[i+3]) > 3:
                        soft_flag = True
                        break
                if soft_flag:
                    self.rtde_c.stopL(a=0.25, asynchronous=True)
                    break
            cnt += 1
            cur_pos_rotvec = self.getPosRotvec()
            diff1 = sum(abs(x-y) for x, y in zip(pos_rotvec[0:3], cur_pos_rotvec[0:3]))
            # 修正-3.14153和3.14152错误判断的情况
            diff2 = sum(min(abs(x-y), abs(x-y-2*math.pi), abs(x-y+2*math.pi)) 
                        for x, y in zip(pos_rotvec[3:6], cur_pos_rotvec[3:6]))
            # print(f'cnt = {cnt}, now_pos = {cur_pos_rotvec}')
            if (diff1 + diff2 < 0.0005) or (time.time() - start_time > 3):
                break
            

    def setPosOrt(self, pos_ort, speed=0.5, acc=0.25, asynchronous=True, 
                    isWait=True, isSoft=False):
        if asynchronous is False:
            # 不是异步通信模式，无法在执行中中断
            isWait = False
            isSoft = False

        rot= utils.rpy2rot_zyx(pos_ort[3], pos_ort[4], pos_ort[5])
        vector= cv2.Rodrigues(rot)[0].tolist()
        # 二维[[1], [2], ..., [n]]转一维[1, 2, ..., n]
        rotvector = [i for item in vector for i in item]
        tcp_pos_rotvec = np.append(pos_ort[0:3], np.array(rotvector))

        self.rtde_c.moveL(tcp_pos_rotvec, speed, acc, asynchronous)
        start_time = time.time()

        cnt = 0
        soft_flag = False
        # 直到完全移动到位，或力/力矩大于阈值，再停止运行此函数
        while isWait:
            if isSoft:
                ft =self.getFTdata()
                for i in range(3):
                    if abs(ft[i]) > 15 or abs(ft[i+3]) > 3:
                        soft_flag = True
                        break
                if soft_flag:
                    # self.rtde_c.stopL(a=0.25, asynchronous=True)
                    pos_ort[2] += 1e-4
                    self.setPosOrt(pos_ort)
                    break
    
            cnt += 1
            cur_pos_rotvec = self.getPosRotvec()
            diff1 = sum(abs(x-y) for x, y in zip(tcp_pos_rotvec[0:3], cur_pos_rotvec[0:3]))
            # 修正-3.14153和3.14152错误判断的情况
            diff2 = sum(min(abs(x-y), abs(x-y-2*math.pi), abs(x-y+2*math.pi), abs(x+y)) 
                        for x, y in zip(tcp_pos_rotvec[3:6], cur_pos_rotvec[3:6]))

            if (diff1 + diff2 < 0.0005) or (time.time() - start_time > 10):
                # print(f'cnt = {cnt}, diff = {diff2}')
                # print(f'tcp = {tcp_pos_rotvec}\n now = {cur_pos_rotvec}')
                # print('------------------------\n')
                break

    def setPosOrtTCP(self, delta_action, speed=0.5, acc=0.25, asynchronous=True, 
                    isWait=True, isSoft=False):
        tcp_pos = self.getPosOrt()
        deltapos_base = utils.tcp2base(tcp_pos, delta_action, sim=False, real=True)
        self.setPosOrt(deltapos_base, speed=speed, acc=acc, asynchronous=asynchronous, 
                       isWait=isWait, isSoft=isSoft)
        
    
    def averageFilter(new_data, buffer=None, window_size=10):
        """
        使用均值滤波处理新的六维力数据。
        
        参数:
        - new_data: 新的六维力数据，类型为numpy数组。
        - buffer: 之前的数据缓冲区，类型为deque，默认为None。
        - window_size: 滤波窗口大小，即用来计算平均值的数据点数量。
        
        返回:
        - 当前窗口内所有数据的平均值，以及更新后的缓冲区。
        """
        if buffer is None:
            buffer = deque(maxlen=window_size)
        buffer.append(new_data)
        
        # 计算当前窗口内的平均值
        filtered_data = np.mean(buffer, axis=0)
        
        return filtered_data, buffer



    def getFTdata(self, isFilter=False):
        if isFilter:
            ft = ftsensor_read.udp_get()
            filtered_data, self.ft_buffer = self.averageFilter(ft, buffer=self.ft_buffer)
            return filtered_data
        else:
            return ftsensor_read.udp_get()
    
    
    def getVisionImage(self):
        return self.camera.get_images()

    
    def openRG2(self):
        self.rob.set_digital_out(0, False)

    def closeRG2(self):
        self.rob.set_digital_out(0, True)

    
