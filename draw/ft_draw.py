import pandas as pd
import matplotlib.pyplot as plt

# 读取csv文件，请确保路径正确
df = pd.read_csv('./data/experimental_data/ft_data.csv') # 根据实际情况调整编码

def draw_xy():
    # 创建图形
    # plt.figure(figsize=(8, 5))

    # 绘制Fx曲线
    plt.plot(df.index, df['Fx'], label='Fx', color='darkorange')

    # 绘制Fy曲线
    plt.plot(df.index, df['Fy'], label='Fy', color='cornflowerblue')

    plt.ylim(-1.5, 1.5)

    # 设置图表标题和坐标轴标签
    # plt.title('Comparison of Fx and Fy over their length index', fontname='Times New Roman', fontsize=14)
    plt.xlabel('Step', fontname='Times New Roman', fontsize=28)
    plt.ylabel('Force (N)', fontname='Times New Roman', fontsize=28)

    # 设置图表中文字体为Times New Roman
    plt.xticks(fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')

    # 显示图例，并设置字体
    plt.legend(prop={'family': 'Times New Roman', 'size': 28})

    # 显示图形
    plt.show()

def draw_z():
    # 创建图形
    # plt.figure(figsize=(8, 5))

    # 绘制Fz曲线
    plt.plot(df.index, df['Fz'], label='Fz', color='limegreen')

    # 设置图表标题和坐标轴标签
    # plt.title('Comparison of Fx and Fy over their length index', fontname='Times New Roman', fontsize=14)
    plt.xlabel('Step', fontname='Times New Roman', fontsize=28)
    plt.ylabel('Force (N)', fontname='Times New Roman', fontsize=28)

    plt.ylim(-16, -10)

    # 设置图表中文字体为Times New Roman
    plt.xticks(fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')

    # 显示图例，并设置字体
    plt.legend(prop={'family': 'Times New Roman', 'size': 28})

    # 显示图形
    plt.show()



if __name__ == "__main__":
    draw_xy()
    # draw_z()