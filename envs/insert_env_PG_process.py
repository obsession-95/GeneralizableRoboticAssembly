# coding: utf-8
import numpy as np
from random import uniform, choice
from collections import deque
import math
import time

SIM = False
REAL = True

RAD2DEG = 180/math.pi

if SIM:
    from robot_control import sim_control
    sim_rob = sim_control.UR5_Sim()

# if REAL:
#     from robot_control import real_rtde
#     self.real_rob = real_rtde.UR5_Rtde(isRealsense=False, isZed=False)

class Insert_Env():
    def __init__(self, baseName, objectName, real_rob=None):
        # 待装配物体
        self.env_name = 'insert'
        self.real_rob = real_rob
        self.objectName = objectName
        if SIM:

            # 待装配物体的底座
            self.baseName = baseName
            self.basePoint = baseName + '_' + self.env_name +'Point'
            self.initialPoint = baseName + '_' + 'search' + 'Point'
            

            self.objectGraspPoint = objectName + '_' + 'graspPoint'
            self.objectAssemblePoint = objectName + '_' + 'assemblePoint'

            # self.initial_pos, _ = sim_rob.getPosOrt(self.initialPoint)

            base_pos, base_ort = sim_rob.getPosOrt(self.basePoint)
            self.base_pos = np.array(base_pos + base_ort)
        
        elif REAL:
            initial_pos_dicts = {
                # TODO According to the physical environment configuration
                # [x, y, z, r, p, y]
            }

            self.initial_pos = initial_pos_dicts[self.objectName]

            base_pos_dicts = {
                # TODO According to the physical environment configuration
                # [x, y, z, r, p, y]
            }
            
            self.base_pos = base_pos_dicts[self.objectName]  


        # 初始化GRU参数
        self.isGRU = True
        self.seq_len = 3
        self.state_seq = deque(maxlen=self.seq_len)     # 存储历史状态

        # 初始化强化学习参数
        self.action_space = np.zeros(6)
        if self.isGRU:
            self.observation_space = np.zeros(14)
        else:
            self.observation_space = np.zeros(15)

        # 初始化环境参数
        self.steps = 0
        self.relative_pos = None
          


    # region gym API
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
    

    def step(self, action=None, isGuide=False):
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
            action = np.zeros(6)

        state = self._get_state()
        direct, ft, relative_pos = state[1:3], state[3:9], state[9:15]
        

        # 策略引导
        if isGuide:
            action_net = self.action_guide(ft, relative_pos, action)
            act = np.copy(action_net)
            a = self.action_transform(act)
        if not isGuide:
            action_net = np.copy(action)
            a = self.action_transform(action)

        # if ft[2] > -0.5 and np.linalg.norm(relative_pos[0:2], ord=np.inf) < 5e-4:

        # delta_a = [a[0], a[1], a[2], 0, 0, 0]
        # 对z轴归一化后的位置误差进行还原
        relative_pz = 100*relative_pos[2]*self.assemble_depth
        s_pz = max(10*relative_pos[2], 0.1)

        if ft[2] > -1:
            s = [direct[0], direct[1], 10, 1, 1, 1] if relative_pz > 5e-3 else [direct[0], direct[1], s_pz, 1, 1, 1]
        else:
            s = [direct[0], direct[1], 1, 1, 1, 1]
            
        delta_action = np.multiply(a, s)

        self.move(delta_action, isTcp=True)

        # time.sleep(0.5)
        state = self._get_state()
        _, ft, relative_pos = state[1:3], state[3:9], state[9:15]


        # 奖励函数
        f_max, t_max = 15, 1
        ft_norm = np.array([ft[0]/f_max, ft[1]/f_max, ft[2]/f_max, 
                            ft[3]/t_max, ft[4]/t_max, ft[5]/t_max])
        
        reward -= 2*np.linalg.norm(ft_norm, ord=2)
        reward -= 5*np.linalg.norm(relative_pos[0:3], ord=2)
        depth_reward = min((1 - 100*relative_pos[2]), 1)     # 装配深度奖励


        if SIM:
            depth = 1e-3
            f_th, t_th = 30, 3
        elif REAL:
            depth = 1e-3
            f_th, t_th = 30, 10

        
        if relative_pos[2] < depth:
            success = True
            done = True
            reward += 150*depth_reward
        
        elif np.linalg.norm(ft[0:3], ord=np.inf) > f_th or np.linalg.norm(ft[3:6], ord=np.inf) > t_th:
            done = True
            reward -= 30*depth_reward
            print(f'F MAX!{np.linalg.norm(ft[0:3], ord=np.inf)}, {np.linalg.norm(ft[3:6], ord=np.inf)}')
        elif np.linalg.norm(relative_pos[0:2], ord=np.inf) > 8e-3:
            done = True
            reward -= 30*depth_reward
            print(f'P MAX!{np.linalg.norm(relative_pos[0:2], ord=np.inf)}')


        if self.isGRU:
            # GRU网络删除当前时间步
            gru_state = np.delete(state, 0)
            # 更新历史状态和动作信息
            self.state_seq.append(gru_state)    # GRU网络删除当前时间步

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

            sim_rob.setObjectParent(self.objectName, 'RG2_attachPoint')   # 将抓取后的物体设置为子对象        
            sim_rob.closeRG2()

            self.grasp_pos, self.grasp_ort = sim_rob.getGraspPosOrt(self.objectName)

            base_pos, base_ort = sim_rob.getPosOrt(self.basePoint)
            move_pos1 = [grasp_pos[0], grasp_pos[1], grasp_pos[2]+0.08]
            sim_rob.moveIK(move_pos1, grasp_ort)
            
            move_pos2 = [base_pos[0], base_pos[1], grasp_pos[2]+0.08]
            sim_rob.moveIK(move_pos2, base_ort)

            

            assemble_pos, assemble_ort = sim_rob.getPosOrt(self.objectAssemblePoint)
            # 修改TCP点位置
            sim_rob.setObjectPosOrt('UR5_tip', assemble_pos, assemble_ort)
            sim_rob.setObjectPosOrt('UR5_target', assemble_pos, assemble_ort)
            
        elif REAL:
            move_pos = np.copy(self.initial_pos)
            move_pos[2] += 0.05

            # self.real_rob.setPosOrt(move_pos)

    
    def pos_detect(self):
        "修正抓取错位情况"
        if SIM:
            sim_rob.setGraspPosOrt(self.objectName, self.grasp_pos, self.grasp_ort)
        elif REAL:
            pass


    def _get_state(self):
        """
        状态空间: 15维向量
                  0: 当前所处时间步
                  1~2: 力引导的方向
                  3~8: 六维力
                  7~14: 六维相对位姿
        """
        ft = self.get_ft()
        ft_label = [self.sign_judge(ft[0]), self.sign_judge(ft[1])]

        relative_pos = self.get_relative_pos()

        self.assemble_depth = self.initial_pos[2] - self.base_pos[2]
        relative_pos[2] = 0.01 * relative_pos[2]/self.assemble_depth

        state = np.concatenate(([self.steps], ft_label, ft, relative_pos), axis=0)

        return state
    

    def get_gru_state(self):
        """
        GRU状态空间
        """
        ft = self.get_ft()
        ft_label = [self.sign_judge(ft[0]), self.sign_judge(ft[1])]

        relative_pos = self.get_relative_pos()

        self.assemble_depth = self.initial_pos[2] - self.base_pos[2]
        relative_pos[2] = 0.01 * relative_pos[2]/self.assemble_depth

        state = np.concatenate((ft_label, ft, relative_pos), axis=0)

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
            ft = self.real_rob.getFTdata()

        return ft
    
    def get_tcp_pos(self):
        "获取TCP点的六维位姿"
        if SIM:
            pos, ort = sim_rob.getPosOrt('UR5_tip')
            tcp_pos = pos + ort   # 合并为一个数组
        elif REAL:
            tcp_pos = self.real_rob.getPosOrt()

        return tcp_pos
    
    
    def get_relative_pos(self):
        "获取相对位置"
        if SIM:
            objectBottom_pos, objectBottom_ort = sim_rob.getPosOrt(self.objectAssemblePoint)
            base_pos, base_ort = sim_rob.getPosOrt(self.basePoint)
            # self.base_pos = np.copy(base_pos + base_ort)

            relative_pos = np.array(objectBottom_pos + objectBottom_ort) - np.array(base_pos + base_ort)
        
        elif REAL:
            tcp_pos = self.real_rob.getPosOrt()
            base_pos = np.copy(self.base_pos)
            relative_pos = np.array(tcp_pos) - np.array(base_pos)

            # print(f'tcp_pos = {tcp_pos}, base_pos = {base_pos}')
            # print(f'relative_pos = {relative_pos}\n')

        
        for i in range(3):
            if abs(relative_pos[i+3]) > 6:
                if SIM:
                    relative_pos[i+3] = objectBottom_ort[i] + base_ort[i]
                elif REAL:
                    relative_pos[i+3] = tcp_pos[i+3] + base_pos[i+3]
        return relative_pos
    

    def move(self, delta_action, isTcp=False):
        # region move
        move_pos = delta_action[0:3]
        move_ort = delta_action[3:6]

        # 基于TCP坐标系
        if isTcp:
            if SIM:
                _, _ = sim_rob.moveSelfIK(move_pos, move_ort, isWait=True, isSoft=True, f_th=1)
            elif REAL:
                self.real_rob.setPosOrtTCP(delta_action, isWait=True, isSoft=True)
        
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
                self.real_rob.setPosOrt(pos_ort, speed=0.5, acc=0.25, asynchronous=True, 
                                isWait=True, isSoft=True)
       

    def _reset(self, err_theta=math.pi, err_r=0):
        "回合重置"
        # region reset
        initial_pos = np.copy(self.base_pos)
        # 引入初始位置误差
        theta = uniform(-err_theta, err_theta)
        r = uniform(0, err_r)

        # initial_pos[0] += r*math.cos(theta)
        # initial_pos[1] += r*math.sin(theta)
        initial_pos[2] += 0.01

        if SIM:
            sim_rob.moveIK(initial_pos[0:3], initial_pos[3:6])
            self.pos_detect()

        elif REAL:
            self.real_rob.setPosOrt(initial_pos, speed=0.5, acc=0.25, asynchronous=False)
        
        state = self._get_state()
        # 对z轴归一化后的位置误差进行还原
        pz_error = 100*state[11]*self.assemble_depth
        initial_pos[2] -= (pz_error - self.assemble_depth)

        if SIM:
            sim_rob.moveIK(initial_pos[0:3], initial_pos[3:6])
        elif REAL:
            self.real_rob.setPosOrt(initial_pos, speed=0.5, acc=0.25, asynchronous=False)

        for _ in range(3):
            if SIM:
                initial_pos[2] -= uniform(3e-4, 5e-4)
                sim_rob.moveIK(initial_pos[0:3], initial_pos[3:6], isSoft=True)
            elif REAL:
                initial_pos[2] -= uniform(8e-4, 12e-4)
                self.real_rob.setPosOrt(initial_pos, speed=0.5, acc=0.25, asynchronous=True, 
                                    isWait=True, isSoft=False)

    
    def action_guide(self, ft, relative_pos, action):
        """
        动作引导
        """
        # x、y动作大小引导
        norm_pos = relative_pos[0:3]*1000
        for i in range(2):
            if abs(norm_pos[i]) > 1:
                action[i] = uniform(0.9, 1.0)
            else:
                action[i] = norm_pos[i]
            # x、y动作方向引导
            if self.sign_judge(ft[i]) != self.sign_judge(action[i]):
                action[i] = -action[i]

        # z方向动作大小引导
        if ft[2] > -0.5:
            # z向下移动
            action[2] = uniform(0.5, 1.0)
        else:
            action[2] = uniform(0, 0.5)

        return action
                       

    def action_transform(self, action, isGuide=False):
        """
        动作空间转换:
            动作空间: [x, y, z, rx, ry, rz]
        """
        # 根据UR5机器人控制精度定义动作空间 平移:0.1mm, 旋转：1°
        # x、y: [0, 5e-4], z: [-1e-3, 5e-3](在step中调整)
        # rx、ry: [-0.5, 0.5], rz:[-1, 1]

         # z方向
        if action[2] < -0.2:
            action[2] = 0
        
        for i in range(2):
            action[i] = (action[i] + 1)/4

        action[0:3] = 5e-4*action[0:3]

        action[3:6] = 0.2*action[3:6]/RAD2DEG
        # action[5] = action[5]/RAD2DEG
        
        for i in range(6):
            if abs(action[i]) < 1e-4:
                action[i] = 0
        
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