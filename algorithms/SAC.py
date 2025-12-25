import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import copy
import os
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAC_countinuous():
    def __init__(self, **kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        self.algorithm_name = 'SAC'


        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim, self.isGRU).to(device)
        self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, self.hidden_dim, self.isGRU).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
        
        self.q_critic_target = copy.deepcopy(self.q_critic)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        # 冻结目标Q网络的参数, 用于平滑更新
        for p in self.q_critic_target.parameters():
            p.requires_grad = False
            
        # 初始化缓存
        self.replay_buffer = ReplayBuffer(max_size=int(1e6), isGRU=self.isGRU, seq_len=self.seq_len)
        # self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e6))

        if self.auto_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = torch.tensor(-self.action_dim, dtype=float, requires_grad=True, device=device)
            # 对alpha取对数，确保alpha始终大于0
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.c_lr)

        print(self.actor)
        print(self.q_critic)

    def select_action(self, state, deterministic):
        # 动作选择关闭梯度计算，只前向传播
        with torch.no_grad():
            '''
            unsqueeze(0)、np.newaxis: 给数组增加一个维度
            将形状为 (n,) 的一维 NumPy 数组转换成形状为 (1, n) 的二维数组
            即创建一个批次大小为1的数据点
            '''
            if self.isGRU:
                # 将历史状态转换为张量并移动到计算设备
                state = torch.FloatTensor(state).unsqueeze(0).to(device)      
            else:
                state = torch.FloatTensor(state[np.newaxis,:]).to(device)
            a, _ = self.actor(state, deterministic, with_logprob=False)

        return a.cpu().numpy()[0]
    
    def sample_action(self):
        a = torch.FloatTensor(self.action_dim).uniform_(-1, 1)
        return a.cpu().numpy()
        

    def train(self,):
        s, a, r, s_next, done = self.replay_buffer.sample(self.batch_size)

        #----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        # 不计算梯度，只计算目标Q值
        with torch.no_grad():
            # actor网络生成下一个状态的动作和对应的对数概率
            a_next, log_pi_a_next = self.actor(s_next, deterministic=False, with_logprob=True)
            # 使用目标Q网络计算下一个状态的Q值
            target_Q1, target_Q2 = self.q_critic_target(s_next, a_next)
            # 双Q物理减少过拟合
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r + (~done) * self.gamma * (target_Q - self.alpha * log_pi_a_next) #Dead or Done is tackled by Randombuffer

        # 计算当前状态的Q值
        current_Q1, current_Q2 = self.q_critic(s, a)

        # 使用均方误差损失函数计算Q网络的损失
        self.q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # 清零梯度
        self.q_critic_optimizer.zero_grad()
        # 反向传播计算梯度
        self.q_loss.backward()
        # 更新Q网络的参数
        self.q_critic_optimizer.step()

        #----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        # 冻结Q网络的参数，防止在更新Actor网络时计算其梯度
        for params in self.q_critic.parameters(): 
            params.requires_grad = False

        # actor网络生成当前状态的动作和对应的对数概率
        a, log_pi_a = self.actor(s, deterministic=False, with_logprob=True)

        # 计算当前状态的Q值
        current_Q1, current_Q2 = self.q_critic(s, a)
        Q = torch.min(current_Q1, current_Q2)

        # 计算Actor网络的损失，目标是最小化负的期望回报加上熵项
        self.a_loss = (self.alpha * log_pi_a - Q).mean()
        self.actor_optimizer.zero_grad()
        self.a_loss.backward()
        self.actor_optimizer.step()

        # 恢复Q网络的参数，允许计算梯度
        for params in self.q_critic.parameters(): 
            params.requires_grad = True

        #----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        if self.auto_alpha:
            # 计算alpha的损失，目标是最小化负的期望熵
            self.alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            self.alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        #----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        # 对目标Q网络的参数进行软更新
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def get_params(self):
        return self.alpha, self.a_loss, self.q_loss,self.alpha_loss


    def save_model(self, save_path):
        torch.save(self.actor.state_dict(), save_path +'/actor.pth')
        torch.save(self.q_critic.state_dict(), save_path +'/q_critic.pth')


    def load_model(self, model_path):
        self.actor.load_state_dict(torch.load("./model/{}/actor.pth".format(model_path), map_location=device))
        self.q_critic.load_state_dict(torch.load("./model/{}/q_critic.pth".format(model_path), map_location=device))


class ReplayBuffer():
    def __init__(self, max_size, isGRU, seq_len):
        self.max_size = max_size    # 最大样本容量
        self.buffer = []            # 缓存
        self.ptr = 0                # 指针
        self.size = 0               # 当前缓存大小

        self.isGRU = isGRU
        self.seq_len = seq_len
    

    def add(self, s, a, r, s_next, done):
        #每次存放一个时间步的数据
        if self.size < self.max_size:
            # 预分配空间
            self.buffer.append(None)

        s = torch.from_numpy(s).float().to(device)
        a = torch.from_numpy(a).float().to(device)
        r = torch.tensor(r, dtype=torch.float, device=device)
        s_next = torch.from_numpy(s_next).float().to(device)
        done = torch.tensor(done, dtype=torch.bool, device=device)

        self.buffer[self.ptr] = (s, a, r, s_next, done)
        self.ptr = int((self.ptr + 1) % self.max_size)  # 环形存储
        self.size = min(self.size + 1, self.max_size)   # 更新缓存大小


    def sample(self, batch_size):
        # 随机选择batch_size个索引
        '''
        torch.randint(): 生成一个包含随机索引的张量，张量的行为batch_size
        '''
        ind = torch.randint(0, self.size, device=device, size=(batch_size,))
        # 提取样本
        batch = [self.buffer[i] for i in ind]

        '''
        batch: 一个列表，其中每个元素是一个包含(s, a, r, s_next, done) 的元组
        zip(*batch): 对于batch解包为独立参数传递给zip函数
                    zip将所有元组中对应位置的元素组合为一个列表，形成一个新的迭代器 
        map(function, iterable): 对参数序列zip(*batch)中每一个元素执行function函数
        lambda x: torch.stack(x): 将每个元素堆叠在一起
        最终, map 函数返回一个迭代器，包含转换后的张量, 通过解包赋值
        '''
        # 将所有张量堆叠在一起
        s, a, r, s_next, done = map(lambda x: torch.stack(x), zip(*batch))
        # 将r、done的shape从[batch_size]转换为[batch_size, 1]
        r = r.unsqueeze(1)
        done = done.unsqueeze(1)
        
        return s, a, r, s_next, done
    

    def save_buffer(self, save_path):
        '''
        将 ReplayBuffer 中的数据保存到 JSON 文件中
        '''
        buffer_path = os.path.join(save_path, 'replay_buffer.json')

        with open(buffer_path, 'w') as f:
            buffer_date = [(s.cpu().numpy().tolist(),
                            a.cpu().numpy().tolist(),
                            r.item(),
                            s_next.cpu().numpy().tolist(), 
                            done.item())
                            for s, a, r, s_next, done in self.buffer]
            json.dump({'ptr': self.ptr, 'size': self.size, 'buffer': buffer_date}, f)


    def load_buffer(self, save_path):
        '''
        从 JSON 文件中加载数据到 ReplayBuffer 中
        '''
        buffer_path = "./model/{}/replay_buffer.json".format(save_path)
        if os.path.exists(buffer_path):
            print('----加载保存的经验回放池数据...')
            with open(buffer_path, 'r') as f:
                data = json.load(f)
                self.ptr = data['ptr']
                self.size = data['size']
                # 将列表转为数组，并恢复原始数据类型
                self.buffer = [(torch.tensor(s, dtype=torch.float, device=device),
                                torch.tensor(a, dtype=torch.float, device=device),
                                torch.tensor(r, dtype=torch.float, device=device),
                                torch.tensor(s_next, dtype=torch.float, device=device),
                                torch.tensor(done, dtype=torch.bool, device=device),)
                               for s, a, r, s_next, done in data['buffer']]
        else:
            print('----初始化经验回放池...')
            self.ptr = 0
            self.size = 0
            self.buffer = []


def build_net(layer_shape, hidden_activation=nn.ReLU, output_activation=nn.ReLU):
    '''
        Build net with for loop
        layer_shape：一个列表，表示每一层的神经元数量
        hidden_activation：隐藏层的激活函数，默认为ReLU
        output_activation：输出层的激活函数，默认为ReLU
    '''
    layers = []
    for j in range(len(layer_shape)-1):
        act = hidden_activation if j < len(layer_shape)-2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, isGRU=False):
        super(Actor, self).__init__()
        self.isGRU = isGRU
        if self.isGRU:
            self.gru_hidden_size = 64
            self.num_layers = 2     # GRU网络层数
            self.gru = nn.GRU(input_size=state_dim, 
                              hidden_size=self.gru_hidden_size, 
                              num_layers=self.num_layers, 
                              batch_first=True, 
                                dropout=0.1)
            # // 除法结果仍为int
            layers = [self.gru_hidden_size] + [hidden_dim, hidden_dim]
            self.a_net = build_net(layers, nn.ReLU, nn.ReLU)

        else:
            # 两个隐藏层
            layers = [state_dim] + [hidden_dim, hidden_dim, hidden_dim]
            self.a_net = build_net(layers, nn.ReLU, nn.ReLU)

        self.mu_layer = nn.Linear(layers[-1], action_dim)
        self.log_std_layer = nn.Linear(layers[-1], action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state, deterministic, with_logprob):
        '''Network with Enforcing Action Bounds'''
        if self.isGRU:
            if not hasattr(self, '_flattened'):
                self.gru.flatten_parameters()
                setattr(self, '_flattened', True)

            gru_out, _ = self.gru(state)
            # self.h_in = h_out
            # 只取最后一个时间步的输出
            net_out = self.a_net(gru_out[:, -1, :])

        else:
            net_out = self.a_net(state)
        # mu: 动作的均值
        mu = self.mu_layer(net_out)
        
        # mu: 动作标准差的对数
        log_std = self.log_std_layer(net_out)
        # 限制标准差对数的范围
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  # 总感觉这里clamp不利于学习
        # we learn log_std rather than std, so that exp(log_std) is always > 0
        std = torch.exp(log_std)
        # 均值为mu，标准差为std的正态分布
        dist = Normal(mu, std)
        # 如果是确定性模式则直接取均值，否则从正态分布中采样
        if deterministic: 
            u = mu
        else: 
            u = dist.rsample()

        '''↓↓↓ Enforcing Action Bounds, see Page 16 of https://arxiv.org/pdf/1812.05905.pdf ↓↓↓'''
        #  将动作限制在[-1, 1]范围内。
        a = torch.tanh(u)
        # 动作的概率密度对数，用于计算熵
        if with_logprob:
            # Get probability density of logp_pi_a from probability density of u:
            # logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
            # Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
            logp_pi_a = dist.log_prob(u).sum(axis=-1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=-1, keepdim=True)
        else:
            logp_pi_a = None

        return a, logp_pi_a


class Double_Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, isGRU=False):
        super(Double_Q_Critic, self).__init__()
        self.isGRU = isGRU
        if self.isGRU:
            self.gru_hidden_size = 64
            self.num_layers = 2     # GRU网络层数
            self.gru_1 = nn.GRU(input_size=state_dim, 
                                hidden_size=self.gru_hidden_size, 
                                num_layers=self.num_layers, 
                                batch_first=True, 
                                dropout=0.1)
            self.gru_2 = nn.GRU(input_size=state_dim, 
                                hidden_size=self.gru_hidden_size, 
                                num_layers=self.num_layers, 
                                batch_first=True, 
                                dropout=0.1)
            layers = [self.gru_hidden_size + action_dim] + [hidden_dim, hidden_dim] + [1]
            # self.h_in_1 = None
            # self.h_in_2 = None
        else:
            # 两个隐藏层
            layers = [state_dim + action_dim] + [hidden_dim, hidden_dim, hidden_dim] + [1]

        # nn.Identity: 不改变输入
        self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)


    def forward(self, state, action):
        # 输入状态和动作
        if self.isGRU:
            if not hasattr(self, '_flattened'):
                self.gru_1.flatten_parameters()
                self.gru_2.flatten_parameters()
                setattr(self, '_flattened', True)

            gru_out_1, _ = self.gru_1(state)
            # self.h_in_1 = h_out_1
            q_in_1 = torch.cat([gru_out_1[:, -1, :], action], dim=-1)   # 只取最后一个时间步的输出
            q1 = self.Q_1(q_in_1)  

            gru_out_2, _ = self.gru_2(state)
            # self.h_in_2 = h_out_2
            q_in_2 = torch.cat([gru_out_2[:, -1, :], action], dim=-1)
            q2 = self.Q_2(q_in_2)

        else:
            sa = torch.cat([state, action], 1)
            # 分别通过两个Q网络计算Q值
            q1 = self.Q_1(sa)
            q2 = self.Q_2(sa)

        return q1, q2