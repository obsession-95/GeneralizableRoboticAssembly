# encoding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np


def ft_draw3():
    # 读取CSV文件的数据
    df = pd.read_csv('./data/experimental_data/Sim SAC Search FT.csv')
    # df = pd.read_csv('./data/experimental_data/Sim SAC Insert FT.csv')

    # 定义列名
    circle_cols = ['circle fx', 'circle fy', 'circle fz', 'circle tx', 'circle ty', 'circle tz']
    six_cols = ['six fx', 'six fy', 'six fz', 'six tx', 'six ty', 'six tz']
    dual_cols = ['dual fx', 'dual fy', 'dual fz', 'dual tx', 'dual fy', 'dual tz']


    # 设置Seaborn的样式，并且确保字体设置不会被覆盖
    sns.set_theme(style='darkgrid', rc={'font.family': 'serif', 
                                        # 'font.serif': ['Times New Roman'], 'font.size': 18})
                                        'font.serif': ['Arial'], 'font.size': 18})

    # 定义颜色
    colors = ['#9d9d9d',    # 灰色
              '#7e99f4',    # 蓝色
              '#78b428',    # 绿色
              '#e4a031']    # 橙色

    linewidth = 1.8  # 线条宽度
    font_size = 20  # 全局字体大小设置

    dimensions = ['Fx(N)', 'Fy(N)', 'Fz(N)', 'Tx(N·m)', 'Ty(N·m)', 'Tz(N·m)']
    objects = ['circle', 'six', 'dual']
    object_labels = {'circle': 'Circular Peg', 'six': 'Hexagonal Peg', 'dual': 'Dual Peg'}
    palette = {'circle': '#7e99f4', 'six': '#78b428', 'dual': '#e4a031'}

    # fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    # for idx, dim in enumerate(dimensions):
    #     row = idx // 3
    #     col = idx % 3

    fig, axes = plt.subplots(1, 6, figsize=(18, 4))
    for idx, dim in enumerate(dimensions):
        ax = axes[idx]
        
        data = {
            'Object': [],
            'Value': []
        }
        
        for obj, cols in zip(objects, [circle_cols, six_cols, dual_cols]):
            if cols[idx] in df.columns:
                data['Object'].extend([obj] * len(df[cols[idx]]))  # 使用原始对象名
                data['Value'].extend(df[cols[idx]])
        
        violin_df = pd.DataFrame(data)
        
        sns.violinplot(x='Object', y='Value', hue='Object',
                       data=violin_df, ax=ax, palette=palette, 
                       inner='box', saturation=1, legend=True,
                    #    density_norm='width',
                       linewidth=1)
        ax.set_xlabel('Objects', fontsize=font_size)
        ax.set_ylabel(f'{dim}', fontsize=font_size)
        ax.set_xticks([])  # 隐藏x轴刻度和标签
        ax.tick_params(axis='y', which='major', labelsize=font_size)

        # 提取第一个子图中的图例句柄以用于创建共享图例
        if idx == 0:
            handles, _ = ax.get_legend_handles_labels()
        ax.legend_.remove()

    # 创建共享图例并放置在图表的顶部
    fig.legend(handles=handles, labels=[object_labels[obj] for obj in objects], loc='upper center', ncol=len(palette), 
               bbox_to_anchor=(0.5, 1.02), 
               columnspacing=6,    # 调整图例之间水平间距
               fontsize=font_size, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("test.png", dpi=500)
    plt.show()


def ft_draw2():
    # 读取CSV文件的数据
    df = pd.read_csv('./data/pic_data/Sim SAC Align FT.csv')  # 此文件仅有12列

    # 定义列名
    # circle_cols = ['circle fx', 'circle fy', 'circle fz', 'circle tx', 'circle ty', 'circle tz']
    six_cols = ['six fx', 'six fy', 'six fz', 'six tx', 'six ty', 'six tz']
    dual_cols = ['dual fx', 'dual fy', 'dual fz', 'dual tx', 'dual fy', 'dual tz']


    # 设置Seaborn的样式，并且确保字体设置不会被覆盖
    sns.set_theme(style='darkgrid', rc={'font.family': 'serif', 
                                        # 'font.serif': ['Times New Roman'], 'font.size': 18})
                                        'font.serif': ['Arial'], 'font.size': 18})

    # 定义颜色
    colors = ['#9d9d9d',    # 灰色
              '#7e99f4',    # 蓝色
              '#78b428',    # 绿色
              '#e4a031']    # 橙色

    linewidth = 1.8  # 线条宽度
    font_size = 20  # 全局字体大小设置

    dimensions = ['Fx(N)', 'Fy(N)', 'Fz(N)', 'Tx(N·m)', 'Ty(N·m)', 'Tz(N·m)']
    objects = ['six', 'dual']
    object_labels = {'six': 'Hexagonal Peg', 'dual': 'Dual Peg'}
    palette = {'six': '#78b428', 'dual': '#e4a031'}

    # fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    # for idx, dim in enumerate(dimensions):
    #     row = idx // 3
    #     col = idx % 3

    fig, axes = plt.subplots(1, 6, figsize=(18, 4))
    for idx, dim in enumerate(dimensions):
        ax = axes[idx]
        
        data = {
            'Object': [],
            'Value': []
        }
        
        for obj, cols in zip(objects, [six_cols, dual_cols]):
            if cols[idx] in df.columns:
                data['Object'].extend([obj] * len(df[cols[idx]]))  # 使用原始对象名
                data['Value'].extend(df[cols[idx]])
        
        violin_df = pd.DataFrame(data)
        
        sns.violinplot(x='Object', y='Value', hue='Object',
                       data=violin_df, ax=ax, palette=palette, 
                       inner='box', saturation=1, legend=True,
                    #    density_norm='width',
                       linewidth=1)
        ax.set_xlabel('Objects', fontsize=font_size)
        ax.set_ylabel(f'{dim}', fontsize=font_size)
        ax.set_xticks([])  # 隐藏x轴刻度和标签
        ax.tick_params(axis='y', which='major', labelsize=font_size)

        # 提取第一个子图中的图例句柄以用于创建共享图例
        if idx == 0:
            handles, _ = ax.get_legend_handles_labels()
        ax.legend_.remove()

    # 创建共享图例并放置在图表的顶部
    fig.legend(handles=handles, labels=[object_labels[obj] for obj in objects], loc='upper center', ncol=len(palette), 
               bbox_to_anchor=(0.5, 1.02), 
               columnspacing=13,    # 调整图例之间水平间距
               fontsize=font_size, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("test.png", dpi=500)
    plt.show()


if __name__ == "__main__":
    ft_draw3()