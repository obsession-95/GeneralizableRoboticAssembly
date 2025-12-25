# coding: utf-8
import numpy as np
from random import uniform, choice
from collections import deque
import math
import torch
from torchvision import transforms
import cv2
import time
from algorithms.MobileNetV3_Lite import MobileNetV3Lite

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SIM = True
REAL = False

RAD2DEG = 180/math.pi

if SIM:
    from robot_control import sim_control
    sim_rob = sim_control.UR5_Sim()

if REAL:
    from robot_control import real_rtde
    real_rob = real_rtde.UR5_Rtde(isRealsense=True, isZed=False)

class Align_Env():
    def __init__(self, baseName, objectName):
        self.env_name = 'align'
        # 待装配物体的底座
        self.baseName = baseName
        self.basePoint = baseName + '_' + self.env_name +'Point'
        
        # 待装配物体
        self.objectName = objectName
        self.objectGraspPoint = objectName + '_' + 'graspPoint'
        self.objectAssemblePoint = objectName + '_' + 'assemblePoint'

        # 初始化GRU参数
        self.isGRU = True
        self.seq_len = 3
        self.state_seq = deque(maxlen=self.seq_len)     # 存储历史状态

        # 初始化强化学习参数
        self.action_space = np.zeros(2)
        if self.isGRU:
            self.observation_space = np.zeros(8)
        else:
            self.observation_space = np.zeros(9)

        # 初始化环境参数
        self.steps = 0
        self.relative_pos = None

        if self.objectName == 'sixpeg' or self.objectName == 'dualpeg':
            self.isBinary = True
        else:
            self.isBinary = False
        # 工件图像处理参数
        # 工件名称: [二值化阈值, 最大面积x, 最小面积x, 最大面积y, 最小面积y]
        # 裁剪：    [裁剪高度上, 高度下, 宽度左, 宽度右]
        if SIM:
            self.img_dicts = {
                'sixpeg':[100, 1828, 17, 1850, 54],
                'dualpeg':[100, 524, 21, 574, 20]
            }

            self.clip_dicts = {
                'sixpeg':[75, 160, 70, 190],
                'dualpeg':[115, 200, 70, 190]
            }

            self.norm_params = {
                # Normalize mean std
                'sixpeg':[0.9725, 0.9725, 0.9725, 0.1574, 0.1574, 0.1574],
                'dualpeg':[0.9725, 0.9725, 0.9725, 0.1574, 0.1574, 0.1574]

            }

            vision_model_path = './model/VisionGuide align/model_align_lite.pth'


        if REAL:
            base_pos_dicts = {
                # TODO According to the physical environment configuration
                # [x, y, z, r, p, y]
                }
            
            
            self.reset_pos = base_pos_dicts[self.objectName]
            # print(self.reset_pos)
            

            self.img_dicts = {
                # TODO Calculated by test_cv.py
            }

            self.clip_dicts = {
                # TODO Crop according to the actual image situation
            }

            self.norm_params = {
                # Normalize mean std
                # TODO Calculated by algorithms.vision_classify.normalize_param()

            }
            vision_model_path = './model/VisionGuide align/model_align_real_' + '.pth'
        
        self.effective_area_x = self.img_dicts[self.objectName][1] - self.img_dicts[self.objectName][2]

        # 初始化视觉分类网络
        model = MobileNetV3Lite(num_classes=2)
        model.load_state_dict(torch.load(vision_model_path))
        self.model = model.eval().to(device)


    def reset(self, theta=math.pi):
        self.steps = 0
        self._reset(err_theta=theta)
        state = self._get_state()

        if self.isGRU:
            # GRU网络删除当前时间步
            gru_state = np.delete(state, 0)
            # 每回合reset时填充初始历史信息
            for _ in range(self.seq_len - 1):
                self.state_seq.append(np.zeros_like(self.observation_space))
            
            self.state_seq.append(gru_state)

            s_seq = np.array(list(self.state_seq))
            
            return s_seq, {}
        else:
            return state, {}
    

    def step(self, action=None, isProcess=False, isCombination=False):
        """
        输入动作，输出状态、奖励
            动作: x、y、z平动
        """
        # region step
        # 初始化
        reward = 0
        done = False
        success = False
        stage = None
        self.steps += 1
        
        if action is None:
            action = np.zeros(4)

        state = self._get_state()
        direct, area, ft = state[1:2], state[2:3], state[3:9]
        # print(f'---state = {state}')

        relative_pos = self.get_relative_pos()

        if ft[2] > -0.5:
            s = [0, 0, 1, 0, 0, 0]
        else:
            s = [self.sign_judge(relative_pos[0]), -self.sign_judge(relative_pos[1]), 0, 0, 0, direct[0]]
        
        if isCombination is False:
            action_net = np.copy(action)
            a = self.action_transform(action)
            if isProcess:
                delta_a = [0, 0, a[0], 0, 0, a[1]]
            else:
                delta_a = [1e-4, 1e-4, a[0], 0, 0, a[1]]
            # print(f'---action = {delta_a}')
        else:
            action_net = None
            delta_a = np.copy(action)

        delta_action = np.multiply(delta_a, s)
        self.move(delta_action, isTcp=True)

        # time.sleep(0.5)
        state = self._get_state()
        _, area, ft = state[1:2], state[2:3], state[3:9]

        tcp_pos = self.get_tcp_pos()
        relative_pos = self.get_relative_pos()

        # 奖励函数
        f_max, t_max = 15, 1
        ft_norm = np.array([ft[0]/f_max, ft[1]/f_max, ft[2]/f_max, 
                            ft[3]/t_max, ft[4]/t_max, ft[5]/t_max])
        
        if area[0] < 0:
            area[0] = 0
        
        reward -= 2*np.linalg.norm(ft_norm, ord=2)     # 接触力奖励
        reward -= 5*(area[0])                          # 用面积表示相对位姿
        # reward += 1000*(self.initial_pz - tcp_pos[2])
        reward -= 2

        if SIM:
            f_th, t_th = 30, 3
            o_th = 8/RAD2DEG
        elif REAL:
            f_th, t_th = 30, 5
            o_th = 13/RAD2DEG

        # if tcp_pos[2] < self.initial_pz - 1e-3:
        if tcp_pos[2] < self.initial_pz - 15e-4:

            success = True
            done = True
            reward += 150
        
        elif np.linalg.norm(ft[0:3], ord=np.inf) > f_th or np.linalg.norm(ft[3:6], ord=np.inf) > t_th:
            done = True
            reward -= 30
            print(f'F MAX!{np.linalg.norm(ft[0:3], ord=np.inf)}, {np.linalg.norm(ft[3:6], ord=np.inf)}')
        elif np.linalg.norm(relative_pos[5:6], ord=np.inf) > o_th:
            done = True
            reward -= 30
            print(f'O MAX!{np.linalg.norm(relative_pos[5:6], ord=np.inf)}')


        if self.isGRU:
            # GRU网络删除当前时间步
            gru_state = np.delete(state, 0)
            # 更新历史状态和动作信息
            self.state_seq.append(gru_state)

            # 防止异常状况，若历史信息长度不足，则用0矩阵填充
            s_seq = [np.zeros_like(self.observation_space)] * (self.seq_len - len(self.state_seq)) + list(self.state_seq)

            return action_net, np.array(s_seq), reward, done, success
        
        else:
            return action_net, state, reward, done, success


    # region 自定义函数
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

            assemble_pos, _ = sim_rob.getPosOrt(self.objectAssemblePoint)
            # 修改TCP点位置
            sim_rob.setObjectPosOrt('UR5_tip', assemble_pos, base_ort)
            sim_rob.setObjectPosOrt('UR5_target', assemble_pos, base_ort)
            
        elif REAL:
            tcp_offset_dicts = {
                'circlepeg':[0, 0, 0, 397.75*1e-3, 0, 0, 0],
                'sixpeg':[0, 0, 0, 397.75*1e-3, 0, 0, 0],
                'dualpeg':[0, 0, 0, 368.00*1e-3, 0, 0, 0]
            }
            
            move_pos = np.copy(self.reset_pos)
            move_pos[2] += 0.05
            # real_rob.setPosOrt(move_pos)

            # 修改TCP点位置
            # real_rob.setTCPOffset(tcp_offset=tcp_offset_dicts[self.objectName])

    
    def action_process(self, action):
        action_net = np.copy(action)
        a = self.action_transform(action_net)
        delta_a = [0, 0, a[0], 0, 0, a[1]]
        # print(f'---action = {delta_a}')

        return delta_a

    
    def pos_detect(self):
        "修正抓取错位情况"
        if SIM:
            # print(f'before:{self.grasp_ort}')
            # self.grasp_ort[0:2] =self.orterr_adjust(self.grasp_ort[0:2])
            # print(f'after:{self.grasp_ort}')
            sim_rob.setGraspPosOrt(self.objectName, self.grasp_pos, self.grasp_ort)
        elif REAL:
            pass


    def _get_state(self):
        """
        状态空间: 9维
                0 当前所处装配时间步
                1:2 从图像中获取的状态信息
                3:8 六维力和力矩
        """
        img_state = self.get_img_state()
        ft = self.get_ft()
        
        state = np.concatenate(([self.steps], img_state, ft), axis=0)

        return state
    

    def get_gru_state(self):
        """
        GRU状态空间
        """
        img_state = self.get_img_state()
        ft = self.get_ft()
        
        state = np.concatenate((img_state, ft), axis=0)

        self.state_seq.append(state)

        # 防止异常状况，若历史信息长度不足，则用0矩阵填充
        s_seq = [np.zeros_like(self.observation_space)] * (self.seq_len - len(self.state_seq)) + list(self.state_seq)

        return np.array(s_seq)





    
    def get_ft(self):
        "获取力/力矩"
        if SIM:
            # 仿真中的力/力矩经过采样周期为10的均值滤波、重力补偿
            ft = sim_rob.gravityCompensate(objectName=self.objectName, isPrint=False)
        elif REAL:
            ft = real_rob.getFTdata()

        return ft
    
    def test_img(self):
        self.first_grasp()
        for _ in range(10):
            self.reset()
            for i in range(50):
                "获取视觉分类标签及阴影部分面积"
                if SIM:
                    clip_param = self.clip_dicts[self.objectName]
                    img_x = sim_rob.getVisionImage(cameraName='Camera_x', clipArr=clip_param)

                elif REAL:
                    img_x, _ = real_rob.getVisionImage()

                    clip_param_x = self.clip_dicts[self.objectName]
                    img_x = img_x[clip_param_x[0]:clip_param_x[1], clip_param_x[2]:clip_param_x[3]]
                    img_x = cv2.resize(img_x, (img_x.shape[1] // 4, img_x.shape[0] // 4))
                    cv2.imshow("img_x", img_x)
                    cv2.waitKey(1000)
                    cv2.destroyAllWindows()

                # 阴影部分面积
                net_img_x, black_area_x = self.img_processing(img_x)

                output = self.model(net_img_x)
                target = output.argmax(1)

                if target == 0:
                    label = -1 
                elif target == 1:
                    label = 1

                if REAL:
                    label *= -1


                print(label)
                act = 1/RAD2DEG
                a = act*label
                delta_a = [0, 0, 0, 0, 0, a]

                self.move(delta_a, isTcp=True)



    def get_img_state(self):
        "获取视觉分类标签及阴影部分面积"
        if SIM:
            clip_param = self.clip_dicts[self.objectName]
            img_x = sim_rob.getVisionImage(cameraName='Camera_x', clipArr=clip_param)
        elif REAL:
            img_x, img_y = real_rob.getVisionImage()

            clip_param_x = self.clip_dicts[self.objectName]
            img_x = img_x[clip_param_x[0]:clip_param_x[1], clip_param_x[2]:clip_param_x[3]]
            img_x = cv2.resize(img_x, (img_x.shape[1] // 4, img_x.shape[0] // 4))
        
        # 阴影部分面积
        net_img_x, black_area_x = self.img_processing(img_x)


        output = self.model(net_img_x)
        target = output.argmax(1)

        if target == 0:
            label = -1 
        elif target == 1:
            label = 1

        if REAL:
            label *= -1
    
        norm_area_x = (black_area_x - self.img_dicts[self.objectName][2])/self.effective_area_x

        img_state = [label, norm_area_x]
        return img_state
            
        
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
    

    def move(self, delta_action, isTcp=False):
        # region move
        move_pos = delta_action[0:3]
        move_ort = delta_action[3:6]

        # 基于TCP坐标系
        if isTcp:
            if SIM:
                _, _ = sim_rob.moveSelfIK(move_pos, move_ort, isWait=True, isSoft=True)
            elif REAL:
                real_rob.setPosOrtTCP(delta_action, isWait=True, isSoft=True)
        
        # 基于基坐标系
        elif not isTcp:
            tcp_pos = self.get_tcp_pos()
        
            tcp_ort = tcp_pos[3:6].copy()
            tcp_ort[0:2] = self.orterr_adjust(tcp_ort[0:2])

            target_pos = list(np.add(tcp_pos[0:3], move_pos))
            target_ort = list(np.add(tcp_pos[3:6], move_ort))
            
            if SIM:
                _, _ = sim_rob.moveIK(target_pos, target_ort, isWait=True, isSoft=True)
            elif REAL:
                pos_ort = target_pos + target_ort
                real_rob.setPosOrt(pos_ort, speed=0.5, acc=0.25, asynchronous=True, 
                                   isWait=True, isSoft=True)


    def _reset(self, err_theta=math.pi, err_r=5e-4):
        "回合重置"
        # region reset
        initial_pos = np.copy(self.reset_pos)
        # 引入初始位置误差

        # r = uniform(1e-3, err_r)
        theta = uniform(-err_theta, err_theta)
        r = uniform(0, err_r)

        initial_pos[0] += r*math.cos(theta)
        initial_pos[1] += r*math.sin(theta)
        initial_pos[2] += 0.01
        initial_pos[5] += choice([-err_theta/30, err_theta/30])

        if SIM:
            sim_rob.moveIK(initial_pos[0:3], initial_pos[3:6])
            self.pos_detect()

        elif REAL:
            real_rob.setPosOrt(initial_pos, speed=0.5, acc=0.25, asynchronous=False)
                
        initial_pos[2] = self.reset_pos[2] + uniform(-3e-4, 3e-4)

        if SIM:
            sim_rob.moveIK(initial_pos[0:3], initial_pos[3:6])
        elif REAL:
            real_rob.setPosOrt(initial_pos, speed=0.5, acc=0.25, asynchronous=False)

        tcp_pos = self.get_tcp_pos()
        self.initial_pz = tcp_pos[2]
        
        # print(f'reset fz = {self.ft[2]}')
                       

    def action_transform(self, action):
        """
        动作空间转换:
            动作空间: [z, rz]
        """
        # 根据UR5机器人控制精度定义动作空间 平移:0.1mm, 旋转：1°
        # x、y: [-1, 1]转换为[0, 1]
        for i in range(2):
            action[i] = (action[i] + 1)/2

            if action[i] < 0.1:
                action[i] = 0
        if action[0] > 0.5:
            # z 平动范围在[0.1, 0.5]之间
            action[0] = 0
        
        action[0] = action[0]*0.001
        action[1] = action[1]/RAD2DEG

        return action

        
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
    

    def img_processing(self, img):
        "对图像进行二值化并计算黑色部分面积"
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if SIM or self.isBinary:
            _, binary_img = cv2.threshold(gray_img, self.img_dicts[self.objectName][0], 255, cv2.THRESH_BINARY)
        elif REAL:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
            clahe_image = clahe.apply(gray_img)
            _, binary_img = cv2.threshold(clahe_image, self.img_dicts[self.objectName][0], 255, cv2.THRESH_BINARY)
        

        # cv2.imshow('img_x', binary_img)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows() # 每次显示完图像后都关闭窗口
        
        
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=self.norm_params[self.objectName][0:3], std=self.norm_params[self.objectName][3:6])
        ])

        if SIM or self.isBinary:
            # 因为模型输入是3通道的，所以把GRAY变成RGB
            rgb_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
            net_img = img_transform(rgb_img)
        elif REAL:
            net_img = img_transform(img)
        # 因为模型输入是3通道的，所以把GRAY变成RGB
        
        net_img = net_img.unsqueeze(0)  # 图片再加一个纬度变成4维
        net_img = net_img.to(device)
    
        # 计算总的像素数量
        total_pixels = binary_img.size
        white_pixel = cv2.countNonZero(binary_img)
        black_pixel = total_pixels - white_pixel
        
        return net_img, black_pixel