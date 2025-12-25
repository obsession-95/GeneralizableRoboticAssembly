# encoding: utf-8
import urx
import math
import cv2
import numpy as np
from robot_control import utils, ftsensor_read, camera_read

RAD2DEG = 180/math.pi

class UR5_Real(object):
    def __init__(self):

        self.rob = urx.Robot("192.168.1.21")

        self.rob.set_tcp((0,0,0.3165,0,0,0))
        self.rob.set_payload(2.0, (0,0,0))
        self.camera = camera_read.CameraReader(isRealsense=True, isZed=True)


    def close(self):
        self.rob.close()
        print("Robot Connection End")
        self.camera.close()
        print("Camera Connection End")

    def getPos(self):
        posture = self.rob.getl()    # 获取位置和姿态

        return posture

    def getRealPosOrient(self):
        posture = self.rob.getl()    # 获取位置和姿态
        # 旋转矢量转欧拉角
        rotvector = np.array(posture[3:6]).reshape(3, 1)
        rot = cv2.Rodrigues(rotvector)[0]
        rpy = utils.rot2euler(rot).tolist()
        
        return posture[0:3], rpy
    
        
    def getJointAngle(self):
        angle = self.rob.getj()

        return angle
    
    
    def setTCPPos(self, pos):
        self.rob.movel(pos, acc=0.25, vel=0.5)

    
    def setPath(self, pos_via, pos_to):
        self.rob.movec(pos_via, pos_to)


    def setTCPPosOrt(self, pos, ort):
        rot = utils.rpy2rot_zyx(ort[0], ort[1], ort[2])
        vector= cv2.Rodrigues(rot)[0].tolist()
        rotvector = [i for item in vector for i in item]

        tcpPos = np.append(pos, np.array(rotvector))
        
        self.rob.movel(tcpPos, acc=0.01, vel=0.1)

    
    def setAdjustPosOrient(self, action):
        posture = self.rob.get_pose()    # 获取初始位置和姿态
        posture.pos[0] += action[0]
        posture.pos[1] += action[1]
        posture.pos[2] += action[2]
        posture.orient.rotate_xb(action[3])     # 设置绕x轴旋转的角度,xb就是x轴
        posture.orient.rotate_yb(action[4])
        posture.orient.rotate_zb(action[5])

        self.rob.set_pose(posture)


    def openRG2(self):
        self.rob.set_digital_out(0, False)

    def closeRG2(self):
        self.rob.set_digital_out(0, True)


    def readFTsensor(self):
        return ftsensor_read.udp_get()
    
    def getVisionImage(self, isRealsense=True, isZed=True):
        if isRealsense and isZed:
            return self.camera.get_images()
        elif isRealsense:
            return self.camera.get_realsense()
        elif isZed:
            return self.camera.get_zed()


if __name__ == "__main__":
    real_robot = UR5_Real()
    # try:
        # for i in range(10):
        #     posture = real_robot.getPosOrient()
        #     print(posture)
    real_robot.openRG2()
    real_robot.closeRG2()
    

    # finally:
    #     real_robot.rob.close()