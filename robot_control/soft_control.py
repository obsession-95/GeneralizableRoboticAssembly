import math
import numpy as np
import utils


class UR5_Soft(object):
    def __init__(self, m=1000, k=200, m1=3000, k1=600, dt=0.02, 
                 xi=0.8, F_th=10, T_th=2, sim=True, real=False):
        self.m = m
        self.k = k      
        self.m1 = m1
        self.k1 = k1

        self.last_epos = np.zeros((6,1))
        self.last_depos = np.zeros((6, 1))
        self.last_ddepos = np.zeros((6, 1))

        # self.b = math.sqrt(m*k)*2*xi
        self.b = xi
        self.b1 = math.sqrt(m1*k1)*2*xi
        self.dt = dt

        self.M = np.array([[self.m, 0, 0, 0, 0, 0],
                           [0, self.m, 0, 0, 0, 0],
                           [0, 0, self.m, 0, 0, 0],
                           [0, 0, 0, self.m1, 0, 0],
                           [0, 0, 0, 0, self.m1, 0],
                           [0, 0, 0, 0, 0, self.m1]])
        
        self.B = np.array([[self.b, 0, 0, 0, 0, 0],
                           [0, self.b, 0, 0, 0, 0],
                           [0, 0, self.b, 0, 0, 0],
                           [0, 0, 0, self.b1, 0, 0],
                           [0, 0, 0, 0, self.b1, 0],
                           [0, 0, 0, 0, 0, self.b1]])
        
        self.K = np.array([[self.k, 0, 0, 0, 0, 0],
                           [0, self.k, 0, 0, 0, 0],
                           [0, 0, self.k, 0, 0, 0],
                           [0, 0, 0, self.k1, 0, 0],
                           [0, 0, 0, 0, self.k1, 0],
                           [0, 0, 0, 0, 0, self.k1]])
        
        self.ft_threshold = np.array([F_th, F_th, -15, T_th, T_th, T_th])
        self.pos_threshold = np.array([0.004, 0.004, 0.004, 0.001, 0.001, 0.001])
        self.vel_threhold = np.array([0.8, 0.8, 0.8, 0.1, 0.1, 0.1])

        self.sim = sim
        self.real = real


    def base2sensor(self, target_pos, now_pos):
        "将导纳控制的输入目标位姿和当前位姿从基坐标系转换到传感器坐标系下"
        FTsensor_height = 0.15
        # 力传感器坐标系在TCP坐标系下的齐次变换矩阵
        T_sensor_nowTCP = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, -FTsensor_height],
                                 [0, 0, 0, 1]])
        # TCP坐标系在力传感器坐标系下的齐次变换矩阵
        T_nowTCP_sensor = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, FTsensor_height],
                                 [0, 0, 0, 1]])
        
        if self.sim:
            target_rot = utils.rpy2rot_xyz(target_pos[3], target_pos[4], target_pos[5])
            now_rot = utils.rpy2rot_xyz(now_pos[3], now_pos[4], now_pos[5])
        elif self.real:
            target_rot = utils.rpy2rot_zyx(target_pos[3], target_pos[4], target_pos[5])
            now_rot = utils.rpy2rot_zyx(now_pos[3], now_pos[4], now_pos[5])
        
        T_targetTCP_base = utils.posort2homogeneous(target_pos[0:3], target_rot)
        T_nowTCP_base = utils.posort2homogeneous(now_pos[0:3], now_rot)
        # 传感器坐标系在基坐标系下的齐次变换矩阵
        self.T_sensor_base = np.dot(T_nowTCP_base, T_sensor_nowTCP)
        # 基坐标系在传感器坐标系下的齐次变换矩阵
        T_base_sensor = np.linalg.inv(self.T_sensor_base)
        # 目标位姿在传感器坐标系下的齐次变换矩阵
        T_targetTCP_sensor = np.dot(T_base_sensor, T_targetTCP_base)

        target_pos_sensor = utils.homogeneous2posort(T_targetTCP_sensor)
        now_pos_sensor = utils.homogeneous2posort(T_nowTCP_sensor)

        return target_pos_sensor, now_pos_sensor
    

    def sensor2base(self, soft_posort):
        "将导纳控制的输出位姿从传感器坐标系转换到基坐标系下"
        if self.sim:
            rot = utils.rpy2rot_xyz(soft_posort[3], soft_posort[4], soft_posort[5])
        elif self.real:
            rot = utils.rpy2rot_zyx(soft_posort[3], soft_posort[4], soft_posort[5])
        
        T_TCP_sensor = utils.posort2homogeneous(soft_posort[0:3], rot)
        T_TCP_base = np.dot(self.T_sensor_base, T_TCP_sensor)
        posort_base = utils.homogeneous2posort(T_TCP_base)

        return posort_base


    def admittance_control(self, target_pos, ft, now_pos):
        """
        导纳控制：
        输入：目标位姿(传感器坐标系下)、接触力、当前位姿(传感器坐标系下)
        输出：调整后的位姿
        """
        target_pos = target_pos.reshape(6, 1)
        ft = ft.reshape(6, 1)
        now_pos = now_pos.reshape(6, 1)
        self.ft_threshold = self.ft_threshold.reshape(6, 1)

        # for i in range(6):
        #     if ft[i] > self.ft_threshold[i]:
        #         ft[i] -= self.ft_threshold[i]
        #     elif ft[i] < -self.ft_threshold[i]:
        #         ft[i] += self.ft_threshold[i]
        #     else:
        #         ft[i] = 0

        # for i in range(3,6):
        #         ft[i] *= 8
        ft[2] = ft[2] - self.ft_threshold[2]

        now_epos = now_pos - target_pos
        for i in range(3,6):
            if abs(now_epos[i][0]) > 6:
                now_epos[i][0] = now_pos[i][0] + target_pos[i][0]

        # ddepos = (ft - self.B * self.last_depos - self.K * now_epos) / self.M

        ddepos = np.dot(np.linalg.inv(self.M), (ft - np.dot(self.B, self.last_depos) 
                                                -np.dot(self.K, now_epos)))
        # if isinstance(ddepos, np.ndarray) and ddepos.ndim == 2 and ddepos.shape == (6, 6):
        #     # 如果 dde 是 6x6 的矩阵，则取主对角线元素
        #     ddepos = ddepos.diagonal()
        
        depos = ddepos*self.dt + self.last_depos

        for i in range(6):
            if depos[i] > 0.8:
                depos[i] = 0.8
            if depos[i] < -0.8:
                depos[i] = -0.8   
        epos = (depos + self.last_depos)/2*self.dt + self.last_epos
        epos_th = 5e-4
        for i in range(3):
            if epos[i] > epos_th:
                epos[i] = epos_th
            if epos[i] < -epos_th:
                epos[i] = -epos_th
        out_pos = epos + target_pos
        
        # self.last_ddepos = ddepos
        # self.last_depos = depos
        # self.last_epos = epos

        # 将输出位姿转为列表
        return out_pos.reshape(-1).tolist()

