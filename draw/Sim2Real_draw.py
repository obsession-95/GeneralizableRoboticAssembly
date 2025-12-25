import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def draw3():
    # 应用Seaborn的主题
    sns.set_theme()

    # 设置全局字体为Times New Roman
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.family'] = 'Arial'

    plt.rcParams['legend.fontsize'] = 20  # 设置图例字体大小

    
    # search
    # success_rate = {'Circular': {'Sim': 100, 'Real': 100}, 
    #                 'Hexagonal': {'Sim': 87, 'Real': 93.3}, 
    #                 'Dual': {'Sim': 100, 'Real': 98.9}}
    # avg_reward = {'Circular': {'Sim': 115.63, 'Real': 105.53},  
    #             'Hexagonal': {'Sim': 70.36, 'Real': 66.07}, 
    #             'Dual': {'Sim': 121.33, 'Real':75.96}}
    # avg_steps = {'Circular': {'Sim': 10.02, 'Real': 10.39}, 
    #             'Hexagonal': {'Sim': 12.46, 'Real': 16.66}, 
    #             'Dual': {'Sim': 7.53, 'Real': 12.13}}

    # insert
    success_rate = {'Circular': {'Sim': 100, 'Real': 100}, 
                    'Hexagonal': {'Sim': 98.8, 'Real': 97.8}, 
                    'Dual': {'Sim': 100, 'Real': 100}}
    avg_reward = {'Circular': {'Sim': 136.58, 'Real': 125.92},  
                'Hexagonal': {'Sim': 137.89, 'Real': 123.28}, 
                'Dual': {'Sim': 131.41, 'Real':115.12}}
    avg_steps = {'Circular': {'Sim': 11.97, 'Real': 9.62}, 
                'Hexagonal': {'Sim': 11.37, 'Real': 8.07}, 
                'Dual': {'Sim': 20.63, 'Real': 8.47}}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5)) 

    bar_width = 0.35
    font_size = 18  # 全局字体大小设置

    def plot_bars(ax, data, title, ylabel):
        n_groups = len(data)
        index = np.arange(n_groups)
        
        real_color = '#7e99f4'   
        sim_color = '#78b428'  
        
        bars_sim = ax.bar(index, [v['Sim'] for v in data.values()], bar_width, label='Sim', color=sim_color, alpha=0.8)
        bars_real = ax.bar(index + bar_width, [v['Real'] for v in data.values()], bar_width, label='Real', color=real_color, alpha=0.8)

        ax.set_xlabel('Object Shape', fontsize=font_size)  
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.set_title(title, fontsize=font_size)
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(data.keys(), fontsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)

        return bars_sim, bars_real

    bars_sim, bars_real = plot_bars(axes[0], success_rate, title=' ', ylabel='Success Rate(%)')
    plot_bars(axes[1], avg_reward, title=' ', ylabel='Average Episode Reward')
    plot_bars(axes[2], avg_steps, title=' ',  ylabel='Average Steps per Episode')

    # 创建一个共享图例，位置设置在图表的顶部中央，横向排列
    fig.legend([bars_sim.patches[0], bars_real.patches[0]], ['Sim', 'Real'], loc='upper center', 
                bbox_to_anchor=(0.5, 1), ncol=2, fontsize=font_size, columnspacing=13,    # 调整图例之间水平间距
                frameon=False)

    # 调整布局以确保图例不会覆盖图表内容，并为图例留出空间
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # 根据需要调整这个值，使图例有足够空间

    plt.savefig("test.png", dpi=500)
    plt.show()


def draw2():
    # 应用Seaborn的主题
    sns.set_theme()

    # 设置全局字体为Times New Roman
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['legend.fontsize'] = 20  # 设置图例字体大小

    
    # align
    success_rate = {'Hexagonal': {'Sim': 87, 'Real': 93.3}, 
                    'Dual': {'Sim': 100, 'Real': 98.9}}
    avg_reward = {'Hexagonal': {'Sim': 70.36, 'Real': 66.07}, 
                  'Dual': {'Sim': 121.33, 'Real':75.96}}
    avg_steps = {'Hexagonal': {'Sim': 12.46, 'Real': 16.66}, 
                 'Dual': {'Sim': 7.53, 'Real': 12.13}}


    fig, axes = plt.subplots(1, 3, figsize=(15, 5)) 

    bar_width = 0.3
    font_size = 18  # 全局字体大小设置

    def plot_bars(ax, data, title, ylabel):
        n_groups = len(data)
        index = np.arange(n_groups)
        
        real_color = '#7e99f4'   
        sim_color = '#78b428'  
        
        bars_sim = ax.bar(index, [v['Sim'] for v in data.values()], bar_width, label='Sim', color=sim_color, alpha=0.8)
        bars_real = ax.bar(index + bar_width, [v['Real'] for v in data.values()], bar_width, label='Real', color=real_color, alpha=0.8)

        ax.set_xlabel('Object Shape', fontsize=font_size)  
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.set_title(title, fontsize=font_size)
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(data.keys(), fontsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)

        return bars_sim, bars_real

    bars_sim, bars_real = plot_bars(axes[0], success_rate, title=' ', ylabel='Success Rate(%)')
    plot_bars(axes[1], avg_reward, title=' ', ylabel='Average Episode Reward')
    plot_bars(axes[2], avg_steps, title=' ',  ylabel='Average Steps per Episode')

    # 创建一个共享图例，位置设置在图表的顶部中央，横向排列
    fig.legend([bars_sim.patches[0], bars_real.patches[0]], ['Sim', 'Real'], loc='upper center', 
                bbox_to_anchor=(0.5, 1), ncol=2, fontsize=font_size, columnspacing=13,    # 调整图例之间水平间距
                frameon=False)

    # 调整布局以确保图例不会覆盖图表内容，并为图例留出空间
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # 根据需要调整这个值，使图例有足够空间

    plt.savefig("test.png", dpi=500)
    plt.show()


if __name__ == "__main__":
    draw3()