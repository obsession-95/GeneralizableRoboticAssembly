import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

def draw():
    # 读取数据
    # loss_df = pd.read_csv('./data/pic_data/MobileNet Search Loss.csv')
    # accuracy_df = pd.read_csv('./data/pic_data/MobileNet Search Accuracy.csv')

    loss_df = pd.read_csv('./data/experimental_data/MobileNet Align Loss.csv')
    accuracy_df = pd.read_csv('./data/experimental_data/MobileNet Align Accuracy.csv')


    train_cols = ['Training (Small)', 'Training (Lite)']
    val_cols = [ 'Validation (Small)', 'Validation (Lite)']



    # 设置Seaborn的样式，并且确保字体设置不会被覆盖
    sns.set_theme(style='darkgrid', rc={'font.family': 'serif', 
                                        # 'font.serif': ['Arial'], 'font.size': 18})
                                        'font.serif': ['Times New Roman'], 'font.size': 18})


    # 定义颜色
    colors = ['#7e99f4',    # 蓝色
              '#78b428']    # 绿色
    
    markers = [' ', ' ']  # 不同的标记符号
    # markers = ['-', '-']  # 不同的标记符号

    linewidth = 1.8  # 线条宽度
    font_size = 18  # 全局字体大小设置
    marker_size = 8

    # 创建图像
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4.5))

    # 绘制Loss图
    for idx, column in enumerate(train_cols):
        ax1.plot(loss_df.index+1, loss_df[column], label=column, 
                 color=colors[idx], marker=markers[idx], markersize=marker_size, markevery=2, linewidth=linewidth)
    ax1.set_xlabel('Epoch', fontsize=font_size)
    ax1.set_ylabel('Training Loss', fontsize=font_size)

    for idx, column in enumerate(val_cols):
        ax2.plot(loss_df.index+1, loss_df[column], label=column, 
                 color=colors[idx], marker=markers[idx], markersize=marker_size, markevery=2, linewidth=linewidth)
    ax2.set_xlabel('Epoch', fontsize=font_size)
    ax2.set_ylabel('Validation Loss', fontsize=font_size)

    # 绘制Accuracy图
    for idx, column in enumerate(train_cols):
        ax3.plot(accuracy_df.index+1, accuracy_df[column], label=column, 
                 color=colors[idx], marker=markers[idx], markersize=marker_size, markevery=2, linewidth=linewidth)
    ax3.set_xlabel('Epoch', fontsize=font_size)
    ax3.set_ylabel('Training Accuracy', fontsize=font_size)

    for idx, column in enumerate(val_cols):
        ax4.plot(accuracy_df.index+1, accuracy_df[column], label=column, 
                 color=colors[idx], marker=markers[idx], markersize=marker_size, markevery=2, linewidth=linewidth)
    ax4.set_xlabel('Epoch', fontsize=font_size)
    ax4.set_ylabel('Validation Accuracy', fontsize=font_size)

    # 共用一个图例
    legend = ['MobileNetV3-Small', 'MobileNetV3-Lite']
    handles, _ = ax1.get_legend_handles_labels()
    fig.legend(handles, legend, loc='upper center', ncol=4, 
               bbox_to_anchor=(0.5, 0.95), 
               columnspacing=13,    # 调整图例之间水平间距
               fancybox=True, shadow=True, prop={'size': font_size}, 
               frameon=False)

    # 设置横纵坐标刻度值的字体大小
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', which='major', labelsize=font_size)
        ax.tick_params(axis='y', which='major', labelsize=font_size)

    # 调整布局以防止图例被截断
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)  # 根据图例的位置调整此值

    # 显示图形
    plt.savefig("test.png", dpi=500)
    plt.show()
    


if __name__ == "__main__":
    draw()