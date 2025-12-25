# encoding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np


def train_draw():
    # 读取数据
    df = pd.read_csv('./data/experimental_data/Sim dual.csv')


    search_cols = ['Search ep_r SAC', 'Search ep_r GRU', 'Search ep_r PG', 'Search ep_r PG GRU']
    align_cols = ['Align ep_r SAC', 'Align ep_r GRU', 'Align ep_r PG', 'Align ep_r PG GRU']
    insert_cols = ['Insert ep_r SAC', 'Insert ep_r GRU', 'Insert ep_r PG', 'Insert ep_r PG GRU']

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

    # 绘制search图
    for idx, col in enumerate(search_cols):
        ax1.plot(df.index+1, df[col], label=col, 
                 color=colors[idx], linewidth=linewidth)

    ax1.set_xlabel('Test Episode', fontsize=font_size)
    ax1.set_ylabel('Episode Reward', fontsize=font_size)
    ax1.set_title('Search', fontsize=font_size)


    # 绘制align图
    for idx, col in enumerate(align_cols):   
        ax2.plot(df.index+1, df[col], label=col, 
                 color=colors[idx], linewidth=linewidth)

    ax2.set_xlabel('Test Episode', fontsize=font_size)
    ax2.set_ylabel('Episode Reward', fontsize=font_size)
    ax2.set_title('Alignment', fontsize=font_size)


    # 绘制insert图
    for idx, col in enumerate(insert_cols):

        ax3.plot(df.index+1, df[col], label=col, 
                 color=colors[idx], linewidth=linewidth)

    ax3.set_xlabel('Test Episode', fontsize=font_size)
    ax3.set_ylabel('Episode Reward', fontsize=font_size)
    ax3.set_title('Insertion', fontsize=font_size)


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
    plt.subplots_adjust(top=0.78)  # 根据图例的位置调整此值

    # 显示图形
    plt.savefig("test.png", dpi=500)
    plt.show()


if __name__ == "__main__":
    train_draw()