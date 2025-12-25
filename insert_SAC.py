from algorithms.utils import str2bool
from algorithms.SAC import SAC_countinuous
from envs.insert_env_PG import Insert_Env
from datetime import datetime
import time
import os
import argparse
import math
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


TRAIN = True
LOG = True
LOADMODEL = False   # 加载模型继续训练

train_episodes = 1000
train_steps = 50
test_episodes = 30
test_steps = 100

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


def main():
    # 初始化
    args = get_args()
    # 初始化环境
    env = Insert_Env(baseName='sixbase', objectName='sixpeg')
    env_name = env.env_name
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.isGRU = env.isGRU
    args.seq_len = env.seq_len
    # 初始化智能体
    agent = SAC_countinuous(**vars(args)) # var: transfer argparse to dictionary  
    algorithm_name = agent.algorithm_name
    
    # 定义
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # 确保卷积操作的确定性，提高实验的可复现性
    torch.backends.cudnn.deterministic = True
    # 禁用 CuDNN 的自动基准测试功能，确保每次运行相同的代码时使用的卷积算法是固定的
    torch.backends.cudnn.benchmark = False

    # region train
    if TRAIN:
        now = datetime.now()
        # 设置时间字符串格式
        timenow = now.strftime("%Y-%m-%d %H-%M")
        folder_path = algorithm_name + env_name + ' ' + timenow

        save_path = './model/' + folder_path
        if not os.path.exists(save_path): 
            os.mkdir(save_path)

        # 实验记录
        if LOG:
            log_path = './log/' + folder_path
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            writer = SummaryWriter(log_dir=log_path)
            writer.add_text("args", str(args))
            writer.add_text("actor_net", str(agent.actor))
            writer.add_text("critic_net", str(agent.q_critic))


        if LOADMODEL:
            model_path = algorithm_name + env_name + ' ' + TRAINEDTIME
            try:
                agent.load_model(model_path)      # 加载保存的神经网络模型
                agent.replay_buffer.load_buffer(model_path)

                print('--Load trained models. Continue training ...\n')
            except Exception as e:
                print('--No trained models. Start first training ...\n')
        
        print('Training......')
        
        env.first_grasp()
        for episodes in range(train_episodes):
            # 初始化装配阶段
            s, info = env.reset()
            episode_reward = 0
            t0 = time.time()

            for steps in range(train_steps):
                if episodes < (args.explore_episodes):
                    act = agent.sample_action()
                else:
                    act = agent.select_action(s, deterministic=False)  # a∈[-1,1]

                a, s_next, r, done, success = env.step(act)
                
                agent.replay_buffer.add(s, a, r, s_next, done)
                s = s_next

                episode_reward += r

                if done:
                    break
            run_time = round(time.time()-t0, 4)

            if agent.replay_buffer.size > args.batch_size:
                # 每回合结束后对agent更新该回合执行步数的3倍
                update_steps = 3*(steps+1)
                for _ in tqdm(range(update_steps), desc='Training Progress', leave=False):
                    agent.train()
            
            success_flag = "SUCCESS" if success else "-------"
            # 显示训练过程
            if agent.replay_buffer.size <= args.batch_size:
                alpha, a_loss, q_loss, alpha_loss = args.alpha, 0, 0, 0
                print(f'Collecting | Episode:{episodes+1}/{train_episodes} | {success_flag} | Step:{steps+1}/{train_steps} | Episode Reward:{round(episode_reward, 4)} | Running Time:{run_time}')
            else:
                alpha, a_loss, q_loss, alpha_loss = agent.get_params()
                print(f'Training | Episode:{episodes+1}/{train_episodes} | {success_flag} | Step:{steps+1}/{train_steps} | Episode Reward:{round(episode_reward, 4)} | Alpha:{round(alpha.item(), 4)} | Running Time:{run_time}')

                '''save model'''
                if episodes % 50 == 0:
                    agent.save_model(save_path)
                    agent.replay_buffer.save_buffer(save_path)

            if LOG:
                writer.add_scalar('ep_r', episode_reward, global_step=episodes)
                writer.add_scalar('step', steps+1, global_step=episodes)
                writer.add_scalar('alpha', alpha, global_step=episodes)
                writer.add_scalar('a_loss', a_loss, global_step=episodes)
                writer.add_scalar('q_loss', q_loss, global_step=episodes)
                writer.add_scalar('alpha_loss', alpha_loss, global_step=episodes)
        agent.save_model(save_path)
        agent.replay_buffer.save_buffer(save_path)

    # region test
    if not TRAIN:
        print('Testing......')
        model_path = algorithm_name + env_name + ' ' + TRAINEDTIME
        agent.load_model(model_path)
        agent.actor.eval()
        agent.q_critic.eval()
        agent.q_critic_target.eval()

        now = datetime.now()
        # 设置时间字符串格式
        timenow = now.strftime("%Y-%m-%d %H-%M")
        folder_path = 'test ' + algorithm_name + env_name + ' ' + timenow

        # 实验记录
        if LOG:
            log_path = './data/' + folder_path
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            writer = SummaryWriter(log_dir=log_path)

        env.first_grasp()
        success_cnt = 0
        total_step = 1

        for episodes in range(test_episodes):
            # 初始化装配阶段
            s, info = env.reset()
            episode_reward = 0
            t0 = time.time()

            for steps in range(test_steps):
                act = agent.select_action(s, deterministic=True)
                a, s_next, r, done, success = env.step(act)

                episode_reward += r
                s = s_next

                if done:
                    if success:
                        success_cnt += 1
                    break

                if LOG:
                    if args.isGRU:
                        writer.add_scalar('fx', s[-1, 2], global_step=total_step + steps)
                        writer.add_scalar('fy', s[-1, 3], global_step=total_step + steps)
                        writer.add_scalar('fz', s[-1, 4], global_step=total_step + steps)
                        writer.add_scalar('tx', s[-1, 5], global_step=total_step + steps)
                        writer.add_scalar('ty', s[-1, 6], global_step=total_step + steps)
                        writer.add_scalar('tz', s[-1, 7], global_step=total_step + steps)
                    else:
                        writer.add_scalar('fx', s[3], global_step=total_step + steps)
                        writer.add_scalar('fy', s[4], global_step=total_step + steps)
                        writer.add_scalar('fz', s[5], global_step=total_step + steps)
                        writer.add_scalar('tx', s[6], global_step=total_step + steps)
                        writer.add_scalar('ty', s[7], global_step=total_step + steps)
                        writer.add_scalar('tz', s[8], global_step=total_step + steps)
            

            run_time = round(time.time()-t0, 4)
            success_flag = "SUCCESS" if success else "-------"
            print(f'Testing | Episode:{episodes+1}/{test_episodes} | {success_flag} | Step:{steps+1}/{test_steps} | Episode Reward:{round(episode_reward, 4)} | Running Time:{run_time}')

            total_step += steps
            if LOG:
                writer.add_scalar('ep_r', episode_reward, global_step=episodes)
                writer.add_scalar('step', steps+1, global_step=episodes)
        
        print(f'Successful Rate = {success_cnt/test_episodes}')


if __name__ == '__main__':
    main()