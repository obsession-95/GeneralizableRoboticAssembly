# encoding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np


def train_draw():
    # 读取数据
    df = pd.read_csv('./data/experimental_data/ft_real_dc.csv')


    f_cols = ['Fx', 'Fy', 'Fz']
    t_cols = ['Tx', 'Ty', 'Tz']


     # 设置Seaborn的样式，并且确保字体设置不会被覆盖
    sns.set_theme(style='darkgrid', rc={'font.family': 'serif', 
                                        # 'font.serif': ['Times New Roman'], 'font.size': 18})
                                        'font.serif': ['Arial'], 'font.size': 18})


    # 定义颜色
    # colors = ['dimgray', '#69af4b', '#7377DD', 'darkorange']
    colors = ['#7e99f4',    # 蓝色
              '#78b428',    # 绿色
              '#e4a031']    # 橙色


    # markers = ['^', '*']  # 不同的标记符号
    linewidth = 1.8  # 线条宽度
    font_size = 18  # 全局字体大小设置

    # 创建图像
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))

    # 绘制force图
    for idx, col in enumerate(f_cols):
        ax1.plot(df.index+1, df[col], label=col, 
                 color=colors[idx], linewidth=linewidth)

    ax1.set_xlabel('Step', fontsize=font_size)
    ax1.set_ylabel('Force (N)', fontsize=font_size)
    ax1.legend(fontsize=font_size, loc='upper center', ncol=len(f_cols), columnspacing=0.8)  # 可以根据需要调整图例的位置
    ax1.set_ylim([-6, 3.5])  # 设置y轴范围，留出一些额外的空间


    # 绘制align图
    for idx, col in enumerate(t_cols):   
        ax2.plot(df.index+1, df[col], label=col, 
                 color=colors[idx], linewidth=linewidth)

    ax2.set_xlabel('Step', fontsize=font_size)
    ax2.set_ylabel('Torque (N·m)', fontsize=font_size)
    ax2.legend(fontsize=font_size, loc='upper center', ncol=len(f_cols), columnspacing=0.8)  # 可以根据需要调整图例的位置
    ax2.set_ylim([-3, 3])  # 设置y轴范围，留出一些额外的空间


    # 设置横纵坐标刻度值的字体大小
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', which='major', labelsize=font_size)
        ax.tick_params(axis='y', which='major', labelsize=font_size)

    # 调整布局以防止图例被截断
    plt.tight_layout()
    # plt.subplots_adjust(top=0.78)  # 根据图例的位置调整此值

    # 显示图形
    plt.savefig("test.png", dpi=500)
    plt.show()


if __name__ == "__main__":
    train_draw()