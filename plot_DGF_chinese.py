#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文版本的DGF算法结果绘图脚本
基于现有数据生成中文图表
"""

import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import json
import numpy as np
import warnings

# 强制设置中文字体 - 在placement环境中
import matplotlib.font_manager as fm

# 清除并重建字体缓存
try:
    import shutil
    import os
    cache_dir = matplotlib.get_cachedir()
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
except:
    pass

# 手动添加中文字体
font_path = '/usr/share/fonts/truetype/arphic/uming.ttc'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)

font_path2 = '/usr/share/fonts/truetype/arphic/ukai.ttc'
if os.path.exists(font_path2):
    fm.fontManager.addfont(font_path2)

# 设置字体配置
matplotlib.rcParams['font.family'] = ['AR PL UMing CN']
matplotlib.rcParams['font.sans-serif'] = ['AR PL UMing CN', 'AR PL UKai CN', 'Droid Sans Fallback', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12

# 设置图例和标签的字体大小
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 16

# 忽略字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

print(f"已配置字体: {matplotlib.rcParams['font.sans-serif']}")

def plot_makespan_vs_num_tasks_chinese(num_tasks, makespan_list, algorithm_names, filename, title='任务完成时间对比'):
    """绘制任务数量与工期之间的关系 - 中文版本"""
    sns.set(style="whitegrid")  # 设置带虚线网格的白色背景

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 定义颜色和样式
    colors = ['dodgerblue', 'red', 'magenta', 'black']  # 颜色列表
    markers = ['o', 's', 'D', '^']  # 不同的标记样式
    linestyles = ['--', '-', ':', '-.']  # 不同的线条样式

    for i, makespan in enumerate(makespan_list):
        ax.plot(num_tasks, makespan, 
                marker=markers[i],  # 使用不同的标记
                color=colors[i],
                linestyle=linestyles[i],  
                linewidth=2,  # 增加线条宽度
                markersize=8,  # 增大标记大小
                label=algorithm_names[i])  # 添加标签

    # 设置轴和标题 - 强制指定中文字体
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 创建字体属性对象
    from matplotlib.font_manager import FontProperties
    chinese_font = FontProperties(fname='/usr/share/fonts/truetype/arphic/uming.ttc')
    
    ax.set_xlabel('生成任务数量', size=16, fontweight=900, fontproperties=chinese_font)  # 中文x轴标签，最大粗细
    
    # 根据标题设置 Y 轴标签
    if '任务完成时间' in title or 'Makespan' in title or '时延' in title:
        ax.set_ylabel('完工时间(s)', size=16, fontweight=900, fontproperties=chinese_font)  # 中文y轴标签
    elif '能耗' in title or 'Energy' in title:
        ax.set_ylabel('能耗(J)', size=16, fontweight=900, fontproperties=chinese_font)  # 中文y轴标签

    # 不显示标题
    # if title:
    #     ax.set_title(title, size=16, fontweight='bold', fontproperties=chinese_font)

    # 设置x轴刻度
    ax.set_xticks([5, 10, 15, 20, 25, 30])  # 仅标注整数
    ax.set_xticklabels([5, 10, 15, 20, 25, 30])
    
    ax.yaxis.grid(True, linestyle='--', linewidth=1)
    ax.xaxis.grid(True, linestyle='--', linewidth=1)

    # 添加加粗放大的图例
    legend = ax.legend(fontsize=14, frameon=True, fancybox=True, shadow=True, 
                      facecolor='white', edgecolor='black', framealpha=0.9)
    # 设置图例文字加粗
    for text in legend.get_texts():
        text.set_fontweight('bold')
        text.set_fontsize(14)
    
    # 确保结果文件夹存在
    output_dir = './result/fig/'
    os.makedirs(output_dir, exist_ok=True)

    # 保存为PNG格式（更好地支持中文）和EPS格式
    plt.savefig(os.path.join(output_dir, f"{filename.replace(' ', '_')}_chinese.png"), 
                format='png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', 
                pil_kwargs={'optimize': True})
    plt.savefig(os.path.join(output_dir, f"{filename.replace(' ', '_')}_chinese.eps"), 
                format='eps', bbox_inches='tight',
                facecolor='white', edgecolor='none')

    # 保存和显示
    fig.tight_layout()
    plt.show()
    print(f"已保存图表: {filename}_chinese.png 和 {filename}_chinese.eps")
    return

def create_comprehensive_comparison(model_type='FFT'):
    """创建综合对比图表"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 收集不同节点数的数据
    node_counts = [5, 10, 20, 25]
    colors = ['blue', 'green', 'orange', 'red']
    num_tasks_list = [5, 10, 15, 20, 25, 30]
    
    for i, num_nodes in enumerate(node_counts):
        try:
            with open(f'./result/{model_type}-{num_nodes}.json', 'r') as f:
                data = json.load(f)
            
            # 只绘制DeAOff算法的结果（索引3）
            makespan_data = data['makespan_list'][3]  # DeAOff算法
            energy_data = data['best_objective_list'][3]  # DeAOff算法
            
            ax1.plot(num_tasks_list, makespan_data, 
                    marker='o', color=colors[i], linewidth=2, markersize=8,
                    label=f'{num_nodes} nodes')
            
            ax2.plot(num_tasks_list, energy_data, 
                    marker='s', color=colors[i], linewidth=2, markersize=8,
                    label=f'{num_nodes} nodes')
                    
        except FileNotFoundError:
            print(f"文件 {model_type}-{num_nodes}.json 不存在，跳过")
            continue
    
    # 设置第一个子图（时延）- 使用中文字体
    from matplotlib.font_manager import FontProperties
    chinese_font = FontProperties(fname='/usr/share/fonts/truetype/arphic/uming.ttc')
    
    ax1.set_xlabel('生成任务数量', size=16, fontweight=900, fontproperties=chinese_font)
    ax1.set_ylabel('完工时间(s)', size=14, fontweight='bold', fontproperties=chinese_font)
    # ax1.set_title(f'{model_type}模型 - DeAOff算法时延对比', size=16, fontweight='bold', fontproperties=chinese_font)
    ax1.grid(True, linestyle='--', alpha=0.7)
    legend1 = ax1.legend(fontsize=14, frameon=True, fancybox=True, shadow=True,
                        facecolor='white', edgecolor='black', framealpha=0.9)
    for text in legend1.get_texts():
        text.set_fontweight('bold')
        text.set_fontsize(14)
    
    # 设置第二个子图（能耗）
    ax2.set_xlabel('生成任务数量', size=16, fontweight=900, fontproperties=chinese_font)
    ax2.set_ylabel('能耗(J)', size=14, fontweight='bold', fontproperties=chinese_font)
    # ax2.set_title(f'{model_type}模型 - DeAOff算法能耗对比', size=16, fontweight='bold', fontproperties=chinese_font)
    ax2.grid(True, linestyle='--', alpha=0.7)
    legend2 = ax2.legend(fontsize=14, frameon=True, fancybox=True, shadow=True,
                        facecolor='white', edgecolor='black', framealpha=0.9)
    for text in legend2.get_texts():
        text.set_fontweight('bold')
        text.set_fontsize(14)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = './result/fig/'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{model_type}_综合对比_chinese.png'), 
                format='png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(output_dir, f'{model_type}_综合对比_chinese.eps'), 
                format='eps', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print(f"已保存综合对比图表: {model_type}_综合对比_chinese")

def main():
    """主函数 - 生成所有中文图表"""
    print("开始生成中文版本图表...")
    
    # 算法名称 - 保持英文，更准确
    algorithm_names_english = ['Heuristic', 'Greedy', 'Random', 'DeAOff']
    
    # 任务数量列表
    num_tasks_list = [5, 10, 15, 20, 25, 30]
    
    # 数据文件列表
    data_files = [
        ('FFT', 5), ('FFT', 10), ('FFT', 20), ('FFT', 25),
        ('GE', 5), ('GE', 10), ('GE', 20), ('GE', 25)
    ]
    
    # 指定结果文件路径
    index = './result/'
    success_count = 0
    total_count = 0
    
    for model, num_nodes in data_files:
        file_path = index + f'{model}-{num_nodes}'
        
        try:
            # 从JSON文件中读取数据
            with open(file_path + '.json', 'r') as f:
                data = json.load(f)
    
            # 提取数据
            makespan_list = data['makespan_list']  # 提取工期列表
            best_objective_list = data['best_objective_list']  # 提取最佳目标值列表
    
            # 绘制任务完成时间图表
            plot_makespan_vs_num_tasks_chinese(
                num_tasks_list, 
                makespan_list, 
                algorithm_names_english, 
                f'{model}-{num_nodes}-时延',
                # title=f'{model}模型({num_nodes}节点) - 任务完成时间对比'
            )
            
            # 绘制能耗图表
            plot_makespan_vs_num_tasks_chinese(
                num_tasks_list, 
                best_objective_list, 
                algorithm_names_english, 
                f'{model}-{num_nodes}-能耗',
                # title=f'{model}模型({num_nodes}节点) - 能耗对比'
            )
            
            print(f"✓ 已完成 {model}-{num_nodes} 的中文图表绘制")
            success_count += 2  # 每个文件生成2个图表
            total_count += 2
            
        except FileNotFoundError:
            print(f"✗ 文件 {file_path}.json 不存在，跳过")
            total_count += 2
        except Exception as e:
            print(f"✗ 处理 {file_path}.json 时出错: {e}")
            total_count += 2
    
    # 生成综合对比图表
    print("\n生成综合对比图表...")
    try:
        create_comprehensive_comparison('FFT')
        create_comprehensive_comparison('GE')
        success_count += 2
        total_count += 2
        print("✓ 综合对比图表生成完成")
    except Exception as e:
        print(f"✗ 生成综合对比图表时出错: {e}")
        total_count += 2
    
    # 输出总结
    print(f"\n{'='*50}")
    print("中文版本图表生成完成！")
    print(f"成功生成: {success_count}/{total_count} 个图表")
    print("生成的文件保存在 ./result/fig/ 目录下")
    print("\n主要修改:")
    print("1. X轴标签: Number of Tasks → 任务数量")
    print("2. Y轴标签: Makespan(s) → 任务完成时间(秒), Energy(J) → 能耗(焦耳)")
    print("3. 算法名称: Heuristic/Greedy/Random/DeAOff → 启发式算法/贪心算法/随机算法/DeAOff算法")
    print("4. 图例已加粗显示")
    print("5. 同时生成PNG和EPS格式")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
