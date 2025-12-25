# coding: utf-8
import numpy as np
from random import uniform, choice
from collections import deque
import math
import matplotlib.pyplot as plt
import cv2
import time
import os

SIM = False
REAL = True

RAD2DEG = 180/math.pi

if SIM:
    from robot_control import sim_control
    sim_rob = sim_control.UR5_Sim()

if REAL:
    from robot_control import real_rtde
    real_rob = real_rtde.UR5_Rtde(isRealsense=True, isZed=True)

class Search_Env():
    def __init__(self, baseName, objectName):
        self.env_name = 'search'
        # 待装配物体的底座
        self.baseName = baseName
        self.basePoint = baseName + '_' + self.env_name +'Point'
        
        # 待装配物体
        self.objectName = objectName
        self.objectGraspPoint = objectName + '_' + 'graspPoint'
        self.objectAssemblePoint = objectName + '_' + 'assemblePoint'

        self.path = os.path.join('./data/vision/real', self.env_name + '_' + self.objectName)
        # self.path = os.path.join('./data/vision', self.env_name + '_' + self.objectName + '1')

        # 工件图像处理参数
        # 工件名称: [二值化阈值, x裁剪高度min, 高度max, 宽度min, 宽度max, y……]
        if SIM:
            self.clip_dicts = {
                'circlepeg':[100, 75, 160, 70, 190],
                'sixpeg':[100, 75, 160, 70, 190],
                'dualpeg':[100, 115, 200, 70, 190]
            }
        if REAL:
            base_pos_dicts = {
                # TODO According to the physical environment configuration
                # [x, y, z, r, p, y]
                }
            
            self.reset_pos = base_pos_dicts[self.objectName]
            # print(self.reset_pos)
            

            self.clip_dicts = {
                # TODO Crop according to the actual image situation
            }

        self.err = 5e-3
        self.zigzag_path()

    
    def first_grasp(self):
        if SIM:
            grasp_pos, grasp_ort = sim_rob.getPosOrt(self.objectGraspPoint)
            sim_rob.moveIK(grasp_pos, grasp_ort)


            # time.sleep(1)
            sim_rob.setObjectParent(self.objectName, 'RG2_attachPoint')   # 将抓取后的物体设置为子对象        
            sim_rob.closeRG2()

            self.grasp_pos, self.grasp_ort = sim_rob.getGraspPosOrt(self.objectName)

            base_pos, base_ort = sim_rob.getPosOrt(self.basePoint)
            move_pos1 = [grasp_pos[0], grasp_pos[1], grasp_pos[2]+0.08]
            sim_rob.moveIK(move_pos1, grasp_ort)
            
            move_pos2 = [base_pos[0], base_pos[1], grasp_pos[2]+0.08]
            sim_rob.moveIK(move_pos2, base_ort)

            self.reset_pos = np.array(base_pos + base_ort)
            
        elif REAL:
            move_pos = np.copy(self.reset_pos)
            move_pos[2] += 0.05
    

    def reset(self):
        "回合重置"
        # region reset
        initial_pos = np.copy(self.reset_pos)

        initial_pos[0] -= self.err
        initial_pos[1] -= self.err

        initial_pos[2] += 0.01

        if SIM:
            sim_rob.moveIK(initial_pos[0:3], initial_pos[3:6])
            self.pos_detect()

            relative_pos = self.get_relative_pos()
            initial_pos[2] -= (relative_pos[2] - uniform(0, 3e-4))
            sim_rob.moveIK(initial_pos[0:3], initial_pos[3:6])

        elif REAL:
            real_rob.setPosOrt(initial_pos, speed=0.5, acc=0.25, asynchronous=False)
        
        while True:
            self.ft = self.get_ft()
            tcp_pos = self.get_tcp_pos()

            if (self.ft[2] < -1) or (tcp_pos[2] < self.reset_pos[2] - 5e-4):
                self.initial_pos = tcp_pos
                self.initial_pz = tcp_pos[2] + 2e-4
                break
            else:
                initial_pos[2] -= 1e-4
                # print(f'---initial_pz = {initial_pos[2]}')

            if SIM:
                sim_rob.moveIK(initial_pos[0:3], initial_pos[3:6], isSoft=True)
            elif REAL:
                real_rob.setPosOrt(initial_pos,  speed=0.5, acc=0.25, asynchronous=True, 
                               isWait=True, isSoft=False)
                
    
    def _move(self, delta_action, isPath=False):
        move_pos = delta_action[0:3]
        move_ort = delta_action[3:6]

        tcp_pos = self.get_tcp_pos()
        
        if isPath:
            tcp_pos[0:2] = self.reset_pos[0:2].copy()
            tcp_pos[2] = self.initial_pz
            tcp_ort = self.initial_pos[3:6].copy()
            tcp_ort = self.orterr_adjust(tcp_ort)


        target_pos = list(np.add(tcp_pos[0:3], move_pos))
        target_ort = list(np.add(tcp_ort, move_ort))

        if SIM:
            _, _ = sim_rob.moveIK(target_pos, target_ort, isWait=True, isSoft=True, f_th=3)
        elif REAL:
            pos_ort = target_pos + target_ort
            real_rob.setPosOrt(pos_ort, speed=0.5, acc=0.25, asynchronous=True, 
                               isWait=True, isSoft=True)
    
    
    def collect_vision_data(self):
        self.first_grasp()
        self.reset()

        path_l = os.path.join(self.path, 'negative/')
        path_r = os.path.join(self.path, 'positive/')
        # path_m = os.path.join(self.path, 'mid/')


        if not os.path.exists(path_l):
            os.makedirs(path_l)
        if not os.path.exists(path_r):
            os.makedirs(path_r)
        # if not os.path.exists(path_m):
        #     os.makedirs(path_m)

        # 初始化数据
        cnt_l_x = 0
        cnt_r_x = 0

        cnt_l_y = 189
        cnt_r_y = 189


        total = len(self.zigzag_x)
        for i in range(total):
            self.pos_detect()

            rz = uniform(-5/RAD2DEG, 5/RAD2DEG)
            dx, dy = self.zigzag_x[i], self.zigzag_y[i]
            delta_action = [dx, dy, 0, 0, 0, rz]

            self._move(delta_action, isPath=True)
            time.sleep(0.5)
            img_x, img_y  = self.get_img()
            # if dx < -5e-4:
            if dx < -1e-3:

                cnt_l_x += 1
                num_l_x = f"{cnt_l_x:04d}"  # 将数字格式化为字符串0001
                cv2.imwrite(path_l + self.objectName + '_' + num_l_x + ".jpg", img_x)
            # elif dx > 5e-4:
            elif dx > 1e-3:
                cnt_r_x += 1
                num_r_x = f"{cnt_r_x:04d}"
                cv2.imwrite(path_r + self.objectName + '_' + num_r_x + ".jpg", img_x)
            # elif dx == 0:    
            #     cnt_m += 1
            #     num_m = f"{cnt_m:04d}"
            #     cv2.imwrite(path_m + self.objectName + '_' + num_m + ".jpg", img)

            if dy > 1e-3:
                cnt_l_y += 1
                num_l_y = f"{cnt_l_y:04d}"  # 将数字格式化为字符串0001
                cv2.imwrite(path_l + self.objectName + '_' + num_l_y + ".jpg", img_y)
            elif dy < -1e-3:
                cnt_r_y += 1
                num_r_y = f"{cnt_r_y:04d}"
                cv2.imwrite(path_r + self.objectName + '_' + num_r_y + ".jpg", img_y)
            
            print(f'Image Collecting | step: {i+1}/{total} | dx = {dx} | dy = {dy}')
        
    
    def zigzag_path(self):
        "之字形轨迹"
        # region 之字形轨迹
        max_err = self.err
        step = 5e-4

        x_min, x_max = -max_err, max_err
        y_min, y_max = -max_err, max_err
        x, y = round(x_min, 4), round(y_min, 4)

        direction = 1   # 1表示向y+方向，-1表示向y-方向
        self.zigzag_x = []
        self.zigzag_y = []


        while x <= x_max:
            self.zigzag_x.append(x)
            self.zigzag_y.append(y)

            if direction == 1:
                if y > y_max - step:
                    x += step
                    direction = -1
                else:
                    y += step
            else:
                if y < y_min + step:
                    x += step
                    direction = 1
                else:
                    y -= step
            x = round(x, 4)
            y = round(y, 4)

        # print(len(self.zigzag_x))
        # plt.figure(figsize=(10, 10))
        # plt.plot(self.zigzag_x, self.zigzag_y, marker='o', linestyle='-')
        # plt.xlim([-6e-3, 6e-3])
        # plt.ylim([-6e-3, 6e-3])
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Zigzag Path')
        # plt.grid(True)
        # plt.show()
    

    def pos_detect(self):
        "修正抓取错位情况"
        if SIM:
            # print(f'before:{self.grasp_ort}')
            # self.grasp_ort[0:2] =self.orterr_adjust(self.grasp_ort[0:2])
            # print(f'after:{self.grasp_ort}')
            sim_rob.setGraspPosOrt(self.objectName, self.grasp_pos, self.grasp_ort)
        elif REAL:
            pass

    
    def get_ft(self):
        "获取力/力矩"
        if SIM:
            # 仿真中的力/力矩经过采样周期为10的均值滤波、重力补偿
            ft = sim_rob.gravityCompensate(objectName=self.objectName, isPrint=False)
        elif REAL:
            ft = real_rob.getFTdata()

        return ft
    

    def get_tcp_pos(self):
        "获取TCP点的六维位姿"
        if SIM:
            pos, ort = sim_rob.getPosOrt('UR5_tip')
            tcp_pos = pos + ort   # 合并为一个数组
        elif REAL:
            tcp_pos = real_rob.getPosOrt()

        return tcp_pos
    

    def get_relative_pos(self):
        "获取相对位置"
        if SIM:
            objectBottom_pos, objectBottom_ort = sim_rob.getPosOrt(self.objectAssemblePoint)
            baseTop_pos, baseTop_ort = sim_rob.getPosOrt(self.basePoint)

            relative_pos = np.array(objectBottom_pos + objectBottom_ort) - np.array(baseTop_pos + baseTop_ort)
        
        elif REAL:
            tcp_pos = real_rob.getPosOrt()
            base_pos = np.copy(self.reset_pos)
            relative_pos = np.array(tcp_pos) - np.array(base_pos)

        # 修正相减后大于2π的情况
        for i in range(3, 6):
            if relative_pos[i] > 6:
                if SIM:
                    relative_pos[i] = objectBottom_ort[i-3] + baseTop_ort[i-3]
                elif REAL:
                    relative_pos[i] = tcp_pos[i] + base_pos[i]
                    
        return relative_pos


    def get_img(self):
        "获取图片"
        if SIM:
            clip_param = self.clip_dicts[self.objectName][1:5]
            img_x = sim_rob.getVisionImage('Camera_x', clip_param)
            img_y = sim_rob.getVisionImage('Camera_y', clip_param)

        elif REAL:
            img_x, img_y = real_rob.getVisionImage()
            
            clip_param_x = self.clip_dicts[self.objectName + '_x']
            img_x = img_x[clip_param_x[0]:clip_param_x[1], clip_param_x[2]:clip_param_x[3]]
            img_x = cv2.resize(img_x, (img_x.shape[1] // 4, img_x.shape[0] // 4))

            clip_param_y = self.clip_dicts[self.objectName + '_y']
            img_y = img_y[clip_param_y[0]:clip_param_y[1], clip_param_y[2]:clip_param_y[3]]
            img_y = cv2.resize(img_y, (img_y.shape[1] // 2, img_y.shape[0] // 2))

        return img_x, img_y


    def sign_judge(self, num):
        "判断参数正负: 正数和0返回1, 负数返回-1"
        if num >= 0:
            return 1
        else:
            return -1


    def orterr_adjust(self, ort):
        "调整姿态误差"
        for i in range(len(ort)):
            if abs(abs(ort[i]) - abs(math.pi)) < 0.2:
                ort[i] = self.sign_judge(ort[i])*math.pi
            elif abs(abs(ort[i]) - abs(math.pi)/2) < 0.2:
                ort[i] = self.sign_judge(ort[i])*math.pi/2
            elif abs(ort[i]) < 0.2:
                    ort[i] = 0

        return ort