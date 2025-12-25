# encoding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np

def smooth(scalars, weight=0.9):
    """
    使用指数移动平均(EMA)对数据进行平滑处理。
    
    参数:
    scalars (list or np.array): 需要平滑的数据序列。
    weight (float): 平滑因子，范围[0,1)，越接近1表示越平滑，默认值为0.6。
    
    返回:
    list: 平滑后的数据序列。
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    
    return smoothed



def train_draw():
    # 读取数据
    # df = pd.read_csv('./data/pic_data/SAC Align.csv')
    # df = pd.read_csv('./data/pic_data/SAC Search.csv')
    df = pd.read_csv('./data/pic_data/SAC Insert.csv')



    epr_cols = ['ep_r', 'ep_r GRU', 'ep_r PG', 'ep_r PG GRU']
    alpha_cols = ['alpha', 'alpha GRU', 'alpha PG', 'alpha PG GRU']
    step_cols = ['step', 'step GRU', 'step PG', 'step PG GRU']


    # 设置Seaborn的样式，并且确保字体设置不会被覆盖
    sns.set_theme(style='darkgrid', rc={'font.family': 'serif', 
                                        # 'font.serif': ['Times New Roman'], 'font.size': 18})
                                        'font.serif': ['Arial'], 'font.size': 18})


    # 定义颜色
    # colors = ['dimgray', '#69af4b', '#7377DD', 'darkorange']
    colors = ['#9d9d9d',    # 灰色
              '#7e99f4',    # 蓝色
              '#78b428',    # 绿色
              '#e4a031']    # 橙色


    # markers = ['^', '*']  # 不同的标记符号
    linewidth = 1.8  # 线条宽度
    font_size = 18  # 全局字体大小设置

    # 创建图像
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # 绘制ep_r图
    for idx, col in enumerate(epr_cols):
        epr_smoothed = smooth(df[col])
        ax1.plot(df.index+1, epr_smoothed, label=col, 
                 color=colors[idx], linewidth=linewidth)
        ax1.fill_between(df.index+1, epr_smoothed - df[col].std(), 
                         epr_smoothed + df[col].std(), color=colors[idx], alpha=0.2)
    ax1.set_xlabel('Training Episode', fontsize=font_size)
    ax1.set_ylabel('Episode Reward', fontsize=font_size)


    # 绘制alpha图
    for idx, col in enumerate(alpha_cols):   
        ax2.plot(df.index+1, df[col], label=col, 
                 color=colors[idx], linewidth=linewidth)
        alpha_smoothed = smooth(df[col])
        # ax2.plot(df.index+1, alpha_smoothed, label=col, 
        #          color=colors[idx], linewidth=linewidth)
        # ax2.fill_between(df.index+1, alpha_smoothed - df[col].std(), 
        #                  alpha_smoothed + df[col].std(), color=colors[idx], alpha=0.2)
    ax2.set_xlabel('Training Episode', fontsize=font_size)
    ax2.set_ylabel('Alpha', fontsize=font_size)


    # 绘制step图
    for idx, col in enumerate(step_cols):
        step_smoothed = smooth(df[col])
        ax3.plot(df.index+1, step_smoothed, label=col, 
                 color=colors[idx], linewidth=linewidth)
        ax3.fill_between(df.index+1, step_smoothed - df[col].std(), 
                         step_smoothed + df[col].std(), color=colors[idx], alpha=0.2)
    ax3.set_xlabel('Training Episode', fontsize=font_size)
    ax3.set_ylabel('Steps', fontsize=font_size)

    # 共用一个图例
    legend = ['SAC', 'SAC + GRU', 'SAC + PG', 'SAC + PG + GRU']
    handles, _ = ax1.get_legend_handles_labels()
    fig.legend(handles, legend, loc='upper center', ncol=4, 
               bbox_to_anchor=(0.5, 0.95), 
               columnspacing=3,    # 调整图例之间水平间距
               fancybox=True, shadow=True, prop={'size': font_size}, frameon=False)

    # 设置横纵坐标刻度值的字体大小
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='x', which='major', labelsize=font_size)
        ax.tick_params(axis='y', which='major', labelsize=font_size)

    # 调整布局以防止图例被截断
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)  # 根据图例的位置调整此值

    # 显示图形
    plt.savefig("test.png", dpi=500)
    plt.show()


if __name__ == "__main__":
    train_draw()