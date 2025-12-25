# -*- coding: utf-8 -*-
from support_files import sim
from robot_control import sim_control
import time
import math
import numpy as np
from random import uniform
# from utils import rpy_to_rotation

sim_rob = sim_control.UR5_Sim()

class  Param_Identify(object):
    def __init__(self, objectName):
        self.objectName = objectName
        self.objectGraspPoint = objectName + '_' + 'graspPoint'
        self.objectAssemblePoint = objectName + '_' + 'assemblePoint'


    def first_grasp(self):
        grasp_pos, grasp_ort = sim_rob.getPosOrt(self.objectGraspPoint)
        sim_rob.moveIK(grasp_pos, grasp_ort)

        sim_rob.closeRG2()
        sim_rob.setObjectParent(self.objectName, 'RG2_attachPoint')   # 将抓取后的物体设置为子对象        
        
        move_pos1 = [grasp_pos[0], grasp_pos[1], grasp_pos[2]+0.13]
        sim_rob.moveIK(move_pos1, grasp_ort)
        
        
    def ikmove(self, initial_pos, initial_ort):
        goal_pos = [0, 0, 0]
        goal_ort = [0, 0, 0]

        for i in range(3):
            goal_pos[i] = initial_pos[i] + uniform(-0.01, 0.01)
            goal_ort[i] = initial_ort[i] + uniform(-0.3, 0.3)

        sim_rob.moveIK(goal_pos, goal_ort)

    
    def run(self):
        sim_rob.setPropertyRespondable('platform')
        sim_rob.setPropertyRespondable('circlebase')
        sim_rob.setPropertyRespondable('sixbase')
        sim_rob.setPropertyRespondable('dualbase')

        F_x = []
        F_y = []
        F_z = []
        M_x = []
        M_y = []
        M_z = []

        cur_pos, cur_ort = sim_rob.getPosOrt('UR5_tip')

        for i in range(30):
            self.ikmove(cur_pos, cur_ort)
            time.sleep(2.0)

            force = [0, 0, 0]
            torque = [0, 0, 0]

            num = 100

            for j in range(num):
                forceRaw, torqueRaw = sim_rob.getFTData()
                force[0] += forceRaw[0]
                force[1] += forceRaw[1]
                force[2] += forceRaw[2]
                torque[0] += torqueRaw[0]
                torque[1] += torqueRaw[1]
                torque[2] += torqueRaw[2]
            
            F_x.append(force[0]/num)
            F_y.append(force[1]/num)
            F_z.append(force[2]/num)
            M_x.append(torque[0]/num)
            M_y.append(torque[1]/num)
            M_z.append(torque[2]/num)

            F_xyz = np.array([[      0,  F_z[i], -F_y[i], 1, 0, 0],
                              [-F_z[i],       0,  F_x[i], 0, 1, 0],
                              [ F_y[i], -F_x[i],       0, 0, 0, 1]])
            M_xyz = np.array([[M_x[i]], [M_y[i]], [M_z[i]]])

            if i == 0:
                F_matrix = F_xyz
                M_matrix = M_xyz
            else:
                F_matrix = np.concatenate((F_matrix, F_xyz), axis=0)
                M_matrix = np.concatenate((M_matrix, M_xyz), axis=0)

        a = np.dot(np.dot(np.linalg.inv(np.dot(F_matrix.T, F_matrix)), F_matrix.T), M_matrix)
        print("A = ", a)
            
        
if __name__ == '__main__':  
    param_identify = Param_Identify()
    param_identify.run()
