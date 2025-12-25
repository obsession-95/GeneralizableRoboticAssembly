from algorithms.utils import str2bool, evaluate_policy
from algorithms.SAC import SAC_countinuous
from search_env_PG_process import Search_Env
from align_env_PG_process import Align_Env
from insert_env_PG_process import Insert_Env
from datetime import datetime
import time
import os
import argparse
import math
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


TRAIN = False
LOG = True
LOADMODEL = False   # 加载训练好的模型

REAL = True

if REAL:
    from robot_control import real_rtde
    real_rob = real_rtde.UR5_Rtde(isRealsense=True, isZed=True)

train_episodes = 1000
train_steps = 50
test_episodes = 30
test_steps = 50

TRAINEDTIME = 'PG GRU'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=28, help='random seed')
    parser.add_argument('--explore_episodes', type=int, default=int(10))
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Parameter soft update coefficient')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden net width, s_dim-400-300-a_dim')
    parser.add_argument('--a_lr', type=float, default=3e-4, help='Learning rate of actor')
    parser.add_argument('--c_lr', type=float, default=3e-4, help='Learning rate of critic')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size of training')
    parser.add_argument('--alpha', type=float, default=1, help='Entropy coefficient')
    parser.add_argument('--auto_alpha', type=str2bool, default=True, help='Use auto_alpha or Not')
    args = parser.parse_args()

    return args


class Search(object):
    def __init__(self, baseName, objectName):
        '''初始化 Search 环境及智能体'''
        # 初始化
        args = get_args()
        self.env = Search_Env(baseName, objectName, real_rob)
        env_name = self.env.env_name
        args.state_dim = self.env.observation_space.shape[0]
        args.action_dim = self.env.action_space.shape[0]
        args.isGRU = self.env.isGRU
        args.seq_len = self.env.seq_len
        # 初始化智能体
        self.agent = SAC_countinuous(**vars(args)) # var: transfer argparse to dictionary  
        algorithm_name = 'SAC'

        model_path = algorithm_name + env_name + ' ' + TRAINEDTIME
        self.agent.load_model(model_path)
        self.agent.actor.eval()
        self.agent.q_critic.eval()
        self.agent.q_critic_target.eval()


    def search_step(self):
        s = self.env.get_gru_state()
        act = self.agent.select_action(s, deterministic=True)
        a, s_next, r, done, success = self.env.step(act, isProcess=True)
        
        return s_next, r, done, success
    

class Align(object):
    def __init__(self, baseName, objectName):
        '''初始化 Align 环境及智能体'''
        # 初始化
        args = get_args()
        self.env = Align_Env(baseName, objectName, real_rob)
        env_name = self.env.env_name
        args.state_dim = self.env.observation_space.shape[0]
        args.action_dim = self.env.action_space.shape[0]
        args.isGRU = self.env.isGRU
        args.seq_len = self.env.seq_len
        # 初始化智能体
        self.agent = SAC_countinuous(**vars(args)) # var: transfer argparse to dictionary  
        algorithm_name = 'SAC'

        model_path = algorithm_name + env_name + ' ' + TRAINEDTIME
        self.agent.load_model(model_path)
        self.agent.actor.eval()
        self.agent.q_critic.eval()
        self.agent.q_critic_target.eval()


    def align_step(self):
        s = self.env.get_gru_state()
        act = self.agent.select_action(s, deterministic=True)
        a, s_next, r, done, success = self.env.step(act, isProcess=True)
        
        return s_next, r, done, success
    

class Insert(object):
    def __init__(self, baseName, objectName):
        '''初始化 Insert 环境及智能体'''
        # 初始化
        args = get_args()
        self.env = Insert_Env(baseName, objectName, real_rob)
        env_name = self.env.env_name
        args.state_dim = self.env.observation_space.shape[0]
        args.action_dim = self.env.action_space.shape[0]
        args.isGRU = self.env.isGRU
        args.seq_len = self.env.seq_len
        # 初始化智能体
        self.agent = SAC_countinuous(**vars(args)) # var: transfer argparse to dictionary  
        algorithm_name = 'SAC'

        model_path = algorithm_name + env_name + ' ' + TRAINEDTIME
        self.agent.load_model(model_path)
        self.agent.actor.eval()
        self.agent.q_critic.eval()
        self.agent.q_critic_target.eval()


    def insert_step(self):
        s = self.env.get_gru_state()
        act = self.agent.select_action(s, deterministic=True)
        a, s_next, r, done, success = self.env.step(act)
        
        return s_next, r, done, success


class SAC_Process2(object):
    def __init__(self, baseName, objectName):
        # 初始化
        args = get_args()
        # np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        # 确保卷积操作的确定性，提高实验的可复现性
        torch.backends.cudnn.deterministic = True
        # 禁用 CuDNN 的自动基准测试功能，确保每次运行相同的代码时使用的卷积算法是固定的
        torch.backends.cudnn.benchmark = False

        # 初始化所有技能
        self.search = Search(baseName, objectName)
        self.insert = Insert(baseName, objectName)
        self.objectName = objectName
        self.algorithm_name = 'SAC'



    def test(self):
        now = datetime.now()
        # 设置时间字符串格式
        timenow = now.strftime("%Y-%m-%d %H-%M")
        folder_path = 'test ' + self.algorithm_name + self.objectName + ' ' + timenow

        # 实验记录
        if LOG:
            log_path = './data/' + folder_path
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            writer = SummaryWriter(log_dir=log_path)

        self.search.env.first_grasp()
        success_cnt = 0
        total_step = 1


        for episodes in range(test_episodes):
            # 初始化装配阶段
            s, info = self.search.env.reset()
            episode_reward = 0
            assembly_stage = 1  # 装配阶段
            
            t0 = time.time()
            for steps in range(test_steps):
                if assembly_stage == 1:     # search
                    s, r, done, success = self.search.search_step()
                    episode_reward += r
                    if done:
                        # search_steps = steps + 1
                        if success:
                            assembly_stage += 1
                            search_final_pos = self.search.env.get_tcp_pos()
                            self.insert.env.initial_pos = np.copy(search_final_pos[0:3])
                            self.insert.env.base_pos[0:2] = search_final_pos[0:2]
                            self.insert.env.base_pos[3:6] = search_final_pos[3:6]
                            print('Search Success !')
                        else:
                            print('Search Failed !')
                            break
                    if LOG:
                        writer.add_scalar('fx', s[-1, 4], global_step=total_step + steps)
                        writer.add_scalar('fy', s[-1, 5], global_step=total_step + steps)
                        writer.add_scalar('fz', s[-1, 6], global_step=total_step + steps)
                        writer.add_scalar('tx', s[-1, 7], global_step=total_step + steps)
                        writer.add_scalar('ty', s[-1, 8], global_step=total_step + steps)
                        writer.add_scalar('tz', s[-1, 9], global_step=total_step + steps)

                elif assembly_stage == 2:     # insert
                    s, r, done, success = self.insert.insert_step()
                    episode_reward += r
                    if done:
                        # insert_steps = steps + 1 - align_steps
                        if success:
                            success_cnt += 1
                            print('Insert Success !')
                        else:
                            print('Search Failed !')
                        break
                    if LOG:
                        writer.add_scalar('fx', s[-1, 2], global_step=total_step + steps)
                        writer.add_scalar('fy', s[-1, 3], global_step=total_step + steps)
                        writer.add_scalar('fz', s[-1, 4], global_step=total_step + steps)
                        writer.add_scalar('tx', s[-1, 5], global_step=total_step + steps)
                        writer.add_scalar('ty', s[-1, 6], global_step=total_step + steps)
                        writer.add_scalar('tz', s[-1, 7], global_step=total_step + steps)

                else:
                    print('Error !')
                    break
            run_time = round(time.time()-t0, 4)
            print(f'Testing | Episode:{episodes+1}/{test_episodes} | Assembly Phase: {assembly_stage} | Step:{steps+1}/{test_steps} | Episode Reward:{round(episode_reward, 4)} | Running Time:{run_time}')

            total_step += steps
            if LOG:
                writer.add_scalar('ep_r', episode_reward, global_step=episodes)
                writer.add_scalar('step', steps+1, global_step=episodes)
                # writer.add_scalar('search_step', search_steps, global_step=episodes)
                # writer.add_scalar('align_step', align_steps, global_step=episodes)
                # writer.add_scalar('insert_step', insert_steps, global_step=episodes)

        print(f'Successful Rate = {success_cnt/test_episodes}')



class SAC_Process3(object):
    def __init__(self, baseName, objectName):
        # 初始化
        args = get_args()
        # np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        # 确保卷积操作的确定性，提高实验的可复现性
        torch.backends.cudnn.deterministic = True
        # 禁用 CuDNN 的自动基准测试功能，确保每次运行相同的代码时使用的卷积算法是固定的
        torch.backends.cudnn.benchmark = False

        # 初始化所有技能
        self.search = Search(baseName, objectName)
        self.align = Align(baseName, objectName)
        self.insert = Insert(baseName, objectName)
        self.objectName = objectName
        self.algorithm_name = 'SAC'



    def test(self):
        now = datetime.now()
        # 设置时间字符串格式
        timenow = now.strftime("%Y-%m-%d %H-%M")
        folder_path = 'test ' + self.algorithm_name + self.objectName + ' ' + timenow

        # 实验记录
        if LOG:
            log_path = './data/' + folder_path
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            writer = SummaryWriter(log_dir=log_path)

        self.search.env.first_grasp()
        success_cnt = 0
        total_step = 1


        for episodes in range(test_episodes):
            # 初始化装配阶段
            s, info = self.search.env.reset()
            episode_reward = 0
            assembly_stage = 1  # 装配阶段
            
            t0 = time.time()
            for steps in range(test_steps):
                if assembly_stage == 1:     # search
                    s, r, done, success = self.search.search_step()
                    episode_reward += r
                    if done:
                        search_steps = steps + 1
                        if success:
                            assembly_stage += 1
                            self.align.env.initial_pz = self.search.env.initial_pz
                            print('Search Success !')
                        else:
                            print('Search Failed !')
                            break
                    if LOG:
                        writer.add_scalar('fx', s[-1, 4], global_step=total_step + steps)
                        writer.add_scalar('fy', s[-1, 5], global_step=total_step + steps)
                        writer.add_scalar('fz', s[-1, 6], global_step=total_step + steps)
                        writer.add_scalar('tx', s[-1, 7], global_step=total_step + steps)
                        writer.add_scalar('ty', s[-1, 8], global_step=total_step + steps)
                        writer.add_scalar('tz', s[-1, 9], global_step=total_step + steps)
    
                elif assembly_stage == 2:   # align
                    s_search = self.search.env.get_gru_state()
                    a_search = self.search.agent.select_action(s_search, deterministic=True)    # 策略网络输出动作：4维
                    act_search = self.search.env.action_process(a_search)   # 机器人执行动作：6维

                    s_align = self.align.env.get_gru_state()
                    a_align = self.align.agent.select_action(s_align, deterministic=True)     # 策略网络输出动作：2维
                    act_align = self.align.env.action_process(a_align)      # 机器人执行动作：6维
                    act = np.concatenate((act_search[0:2], act_align[2:6]), axis=0)

                    _, s, r, done, success = self.align.env.step(act, isCombination=True)
                    episode_reward += r
                    if done:
                        align_steps = steps  + 1 - search_steps
                        if success:
                            assembly_stage += 1
                            align_final_pos = self.align.env.get_tcp_pos()
                            self.insert.env.initial_pos = np.copy(align_final_pos[0:3])
                            self.insert.env.base_pos[0:2] = align_final_pos[0:2]
                            self.insert.env.base_pos[3:6] = align_final_pos[3:6]

                            print('Align Success !')
                        else:
                            print('Align Failed !')
                            break
                    if LOG:
                        writer.add_scalar('fx', s[-1, 2], global_step=total_step + steps)
                        writer.add_scalar('fy', s[-1, 3], global_step=total_step + steps)
                        writer.add_scalar('fz', s[-1, 4], global_step=total_step + steps)
                        writer.add_scalar('tx', s[-1, 5], global_step=total_step + steps)
                        writer.add_scalar('ty', s[-1, 6], global_step=total_step + steps)
                        writer.add_scalar('tz', s[-1, 7], global_step=total_step + steps)

                elif assembly_stage == 3:     # insert
                    
                    s, r, done, success = self.insert.insert_step()
                    episode_reward += r
                    if done:
                        insert_steps = steps + 1 - align_steps
                        if success:
                            success_cnt += 1
                            print('Insert Success !')
                        else:
                            print('Search Failed !')
                        break
                    if LOG:
                        writer.add_scalar('fx', s[-1, 2], global_step=total_step + steps)
                        writer.add_scalar('fy', s[-1, 3], global_step=total_step + steps)
                        writer.add_scalar('fz', s[-1, 4], global_step=total_step + steps)
                        writer.add_scalar('tx', s[-1, 5], global_step=total_step + steps)
                        writer.add_scalar('ty', s[-1, 6], global_step=total_step + steps)
                        writer.add_scalar('tz', s[-1, 7], global_step=total_step + steps)

                else:
                    print('Error !')
                    break
            run_time = round(time.time()-t0, 4)
            print(f'Testing | Episode:{episodes+1}/{test_episodes} | Assembly Phase: {assembly_stage} | Step:{steps+1}/{test_steps} | Episode Reward:{round(episode_reward, 4)} | Running Time:{run_time}')

            total_step += steps
            if LOG:
                writer.add_scalar('ep_r', episode_reward, global_step=episodes)
                writer.add_scalar('step', steps+1, global_step=episodes)
                # writer.add_scalar('search_step', search_steps, global_step=episodes)
                # writer.add_scalar('align_step', align_steps, global_step=episodes)
                # writer.add_scalar('insert_step', insert_steps, global_step=episodes)

        print(f'Successful Rate = {success_cnt/test_episodes}')


class SAC_Process4(object):
    def __init__(self, baseName, objectName):
        # 初始化
        args = get_args()
        # np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        # 确保卷积操作的确定性，提高实验的可复现性
        torch.backends.cudnn.deterministic = True
        # 禁用 CuDNN 的自动基准测试功能，确保每次运行相同的代码时使用的卷积算法是固定的
        torch.backends.cudnn.benchmark = False

        # 初始化所有技能
        self.search = Search(baseName, objectName)
        self.align = Align(baseName, objectName)
        self.insert = Insert(baseName, objectName)
        self.objectName = objectName
        self.algorithm_name = 'SAC'



    def test(self):
        now = datetime.now()
        # 设置时间字符串格式
        timenow = now.strftime("%Y-%m-%d %H-%M")
        folder_path = 'test ' + self.algorithm_name + self.objectName + ' ' + timenow

        # 实验记录
        if LOG:
            log_path = './data/' + folder_path
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            writer = SummaryWriter(log_dir=log_path)

        self.search.env.first_grasp()
        success_cnt = 0
        total_step = 1


        for episodes in range(test_episodes):
            # 初始化装配阶段
            s, info = self.search.env.reset()
            episode_reward = 0
            assembly_stage = 1  # 装配阶段
            
            t0 = time.time()
            for steps in range(test_steps):
                if assembly_stage == 1:     # search
                    s, r, done, success = self.search.search_step()
                    episode_reward += r
                    if done:
                        # search_steps = steps + 1
                        if success:
                            assembly_stage += 1
                            search_final_pos = self.search.env.get_tcp_pos()
                            self.insert.env.initial_pos = np.copy(search_final_pos[0:3])
                            self.insert.env.base_pos[0:2] = search_final_pos[0:2]
                            self.insert.env.base_pos[3:6] = search_final_pos[3:6]
                            self.insert.env.base_pos[2] = 66.33*1e-3
                            self.align.env.initial_pz = self.search.env.initial_pz
                            print('Search Success !')
                        else:
                            print('Search Failed !')
                            break
                    if LOG:
                        writer.add_scalar('fx', s[-1, 4], global_step=total_step + steps)
                        writer.add_scalar('fy', s[-1, 5], global_step=total_step + steps)
                        writer.add_scalar('fz', s[-1, 6], global_step=total_step + steps)
                        writer.add_scalar('tx', s[-1, 7], global_step=total_step + steps)
                        writer.add_scalar('ty', s[-1, 8], global_step=total_step + steps)
                        writer.add_scalar('tz', s[-1, 9], global_step=total_step + steps)

                elif assembly_stage == 2:     # insert
                    s, r, done, success = self.insert.insert_step()
                    episode_reward += r
                    if done:
                        # insert_steps = steps + 1 - align_steps
                        if success:
                            success_cnt += 1
                            print('Insert Success !')
                        else:
                            print('Search Failed !')
                        break
                    if LOG:
                        writer.add_scalar('fx', s[-1, 2], global_step=total_step + steps)
                        writer.add_scalar('fy', s[-1, 3], global_step=total_step + steps)
                        writer.add_scalar('fz', s[-1, 4], global_step=total_step + steps)
                        writer.add_scalar('tx', s[-1, 5], global_step=total_step + steps)
                        writer.add_scalar('ty', s[-1, 6], global_step=total_step + steps)
                        writer.add_scalar('tz', s[-1, 7], global_step=total_step + steps)
    
                elif assembly_stage == 3:   # align
                    s, r, done, success = self.align.align_step()
                    episode_reward += r
                    if done:
                        # align_steps = steps  + 1 - search_steps
                        if success:
                            assembly_stage += 1
                            align_final_pos = self.align.env.get_tcp_pos()
                            self.insert.env.initial_pos = np.copy(align_final_pos[0:3])
                            self.insert.env.base_pos[0:2] = align_final_pos[0:2]
                            self.insert.env.base_pos[3:6] = align_final_pos[3:6]
                            self.insert.env.base_pos[2] = 45*1e-3

                            print('Align Success !')
                        else:
                            print('Align Failed !')
                            break
                    if LOG:
                        writer.add_scalar('fx', s[-1, 2], global_step=total_step + steps)
                        writer.add_scalar('fy', s[-1, 3], global_step=total_step + steps)
                        writer.add_scalar('fz', s[-1, 4], global_step=total_step + steps)
                        writer.add_scalar('tx', s[-1, 5], global_step=total_step + steps)
                        writer.add_scalar('ty', s[-1, 6], global_step=total_step + steps)
                        writer.add_scalar('tz', s[-1, 7], global_step=total_step + steps)

                elif assembly_stage == 4:     # insert
                    
                    s, r, done, success = self.insert.insert_step()
                    episode_reward += r
                    if done:
                        # insert_steps = steps + 1 - align_steps
                        if success:
                            success_cnt += 1
                            print('Insert Success !')
                        else:
                            print('Search Failed !')
                        break
                    if LOG:
                        writer.add_scalar('fx', s[-1, 2], global_step=total_step + steps)
                        writer.add_scalar('fy', s[-1, 3], global_step=total_step + steps)
                        writer.add_scalar('fz', s[-1, 4], global_step=total_step + steps)
                        writer.add_scalar('tx', s[-1, 5], global_step=total_step + steps)
                        writer.add_scalar('ty', s[-1, 6], global_step=total_step + steps)
                        writer.add_scalar('tz', s[-1, 7], global_step=total_step + steps)

                else:
                    print('Error !')
                    break
            run_time = round(time.time()-t0, 4)
            print(f'Testing | Episode:{episodes+1}/{test_episodes} | Assembly Phase: {assembly_stage} | Step:{steps+1}/{test_steps} | Episode Reward:{episode_reward} | Running Time:{run_time}')

            total_step += steps
            if LOG:
                writer.add_scalar('ep_r', episode_reward, global_step=episodes)
                writer.add_scalar('step', steps+1, global_step=episodes)
                # writer.add_scalar('search_step', search_steps, global_step=episodes)
                # writer.add_scalar('align_step', align_steps, global_step=episodes)
                # writer.add_scalar('insert_step', insert_steps, global_step=episodes)

        print(f'Successful Rate = {success_cnt/test_episodes}')


if __name__ == '__main__':
    main = SAC_Process2(baseName='dcbase', objectName='dc')
    # main = SAC_Process3(baseName='usbbase', objectName='usb')
    # main = SAC_Process4(baseName='flangebase', objectName='flange')

    main.test()