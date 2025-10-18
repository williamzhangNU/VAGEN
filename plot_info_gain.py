"""
简化的信息增益可视化脚本

只生成两张图：
1. Active Text 配置的信息增益对比
2. Active Vision 配置的信息增益对比

用法:
    python plot_info_gain_simple.py
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, List, Union
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

def read_info_gain_from_models(results_dir: str) -> tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, List[int]]]]:
    """读取不同模型目录下的env_data.json文件，提取信息增益值和样本结束步数（仅active配置）

    Returns:
        tuple: (info_gains, sample_end_steps)
        - info_gains: 模型名 -> 配置名 -> 信息增益列表
        - sample_end_steps: 模型名 -> 配置名 -> 样本结束步数列表
    """
    info_gains = {}
    sample_end_steps = {}

    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

        env_data_path = os.path.join(model_path, "env_data.json")
        if not os.path.exists(env_data_path):
            continue

        try:
            with open(env_data_path, 'r') as f:
                data = json.load(f)

            model_name = data.get('meta_info', {}).get('model_name', model_dir)
            exp_summary = data.get('exp_summary', {}).get('group_performance', {})
            samples = data.get('samples', {})

            model_configs = {}
            model_end_steps = {}

            # 从聚合数据中获取信息增益
            for config_name, config_data in exp_summary.items():
                # 只包含active配置
                if "active" not in config_name.lower():
                    continue

                infogain_per_turn = config_data.get('infogain_per_turn')
                if infogain_per_turn is not None and isinstance(infogain_per_turn, list) and len(infogain_per_turn) > 0:
                    model_configs[config_name] = infogain_per_turn
                else:
                    # 回退到使用avg_final_information_gain作为单个值
                    final_info_gain = config_data.get('avg_final_information_gain')
                    if final_info_gain is not None:
                        model_configs[config_name] = [final_info_gain]  # 转换为列表格式

            for config_name in model_configs.keys():
                end_steps = []
                for sample_name, sample_data in samples.items():
                    if config_name in sample_data:
                        config_sample_data = sample_data[config_name]
                        if isinstance(config_sample_data, dict) and 'env_turn_logs' in config_sample_data:
                            turn_logs = config_sample_data['env_turn_logs']
                            if turn_logs and len(turn_logs) > 0:
                                # 获取最后一个turn的turn_number作为结束步数
                                last_turn_number = turn_logs[-1].get('turn_number', 0)
                                end_steps.append(last_turn_number)

                if end_steps:
                    model_end_steps[config_name] = end_steps

            if model_configs:
                info_gains[model_name] = model_configs
                sample_end_steps[model_name] = model_end_steps
                print(f"找到 {model_name} 的信息增益: {list(model_configs.keys())}")
                print(f"  样本结束步数: {[(config, len(steps)) for config, steps in model_end_steps.items()]}")

        except Exception as e:
            print(f"读取 {env_data_path} 出错: {e}")

    return info_gains, sample_end_steps


def plot_config_group_line_style(info_gains: Dict[str, Dict[str, List[float]]],
                                sample_end_steps: Dict[str, Dict[str, List[int]]],
                                group_name: str,
                                group_configs: list,
                                save_path: str) -> None:
    """为配置组创建折线图风格的图表（参考提供的图像样式）"""
    if not group_configs:
        print(f"没有找到 {group_name} 的配置")
        return

    # 准备数据 - 为每个配置创建单独的线
    plt.figure(figsize=(8, 4))

    # 定义颜色和标记样式
    colors = ['#4285f4', '#ea4335', '#34a853', '#fbbc05', '#ff6d00', '#673ab7']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    linestyles = ['-', '--', ':', '-.']

    # 收集所有模型的数据，确定最大turn数
    max_turns = 0

    # 收集该配置组中所有模型的数据
    model_data = {}
    for model_name, model_configs in info_gains.items():
        for config in group_configs:
            if config in model_configs:
                infogain_list = model_configs[config]
                # 为每个模型-配置组合创建唯一标识
                model_config_key = f"{model_name}_{config}"
                model_data[model_config_key] = {
                    'model_name': model_name,
                    'config': config,
                    'data': infogain_list
                }
                max_turns = max(max_turns, len(infogain_list))

    if max_turns == 0:
        print(f"没有数据可绘制 {group_name}")
        return

    # x轴为step数（从1开始）
    x_positions = range(1, max_turns + 1)

    # 为每个模型-配置组合绘制线条
    line_index = 0
    legend_labels = []
    model_colors = {}  # 存储每个模型对应的颜色

    for model_config_key, data_info in model_data.items():
        model_name = data_info['model_name']
        config = data_info['config']
        infogain_list = data_info['data']

        if infogain_list:
            # 创建图例标签：只显示模型名
            model_short = model_name.split('/')[-1] if '/' in model_name else model_name

            # 绘制线和点，只在奇数步数显示标记
            line_color = colors[line_index % len(colors)]

            # 创建数据点，只在奇数位置显示标记
            x_data = x_positions[:len(infogain_list)]
            y_data = infogain_list

            # 绘制线条（所有点）
            plt.plot(x_data, y_data,
                    color=line_color,
                    linestyle=linestyles[line_index % len(linestyles)],
                    linewidth=2.5,
                    alpha=0.8,
                    label=model_short)

            # 只在奇数步数位置显示标记
            for i, (x, y) in enumerate(zip(x_data, y_data)):
                if x % 2 == 1:  # 奇数步数
                    plt.plot(x, y,
                           marker=markers[line_index % len(markers)],
                           color=line_color,
                           markersize=5,
                           markerfacecolor='white',
                           markeredgewidth=2)

            legend_labels.append(model_short)

            # 存储模型对应的颜色
            model_colors[model_name] = line_color

            line_index += 1

    # 自定义图表样式
    plt.title(f'Accumulated Information Gain', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Steps', fontsize=14, fontweight='bold')
    plt.ylabel('Information Gain', fontsize=14, fontweight='bold')

    # 设置x轴标签 - 每2步显示一个标签，统一字体大小
    x_ticks = [x for x in x_positions if x % 2 == 1]  # 显示1, 3, 5, 7...
    plt.xticks(x_ticks, fontsize=14)
    plt.yticks(fontsize=14)

    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')

    # 绘制50%样本结束步数的竖线
    vline_positions = {}  # 用于跟踪每个步数位置的竖线数量
    vline_added_to_legend = False  # 确保只添加一次到图例

    for model_name, model_end_steps in sample_end_steps.items():
        for config in group_configs:
            if config in model_end_steps:
                end_steps_list = model_end_steps[config]
                if end_steps_list:
                    # 计算50%分位数（中位数）
                    end_steps_sorted = sorted(end_steps_list)
                    median_step = np.percentile(end_steps_sorted, 80)

                    # 计算偏移量，避免重叠
                    if median_step in vline_positions:
                        vline_positions[median_step] += 1
                        offset = (vline_positions[median_step] - 1) * 0.1 - 0.05  # 左右错开
                    else:
                        vline_positions[median_step] = 1
                        offset = 0

                    # 使用与模型线条相同的颜色绘制竖线
                    vline_color = model_colors.get(model_name, 'gray')

                    # 第一条竖线添加到图例中，图例中使用灰色
                    if not vline_added_to_legend:
                        plt.axvline(x=median_step + offset,
                                  color=vline_color,
                                  linestyle='--',
                                  alpha=0.7,
                                  linewidth=2)
                        # 单独创建一个灰色的图例项
                        plt.plot([], [], color='gray', linestyle='--',
                               linewidth=2, label='80% Samples Ends')
                        vline_added_to_legend = True
                    else:
                        plt.axvline(x=median_step + offset,
                                  color=vline_color,
                                  linestyle='--',
                                  alpha=0.7,
                                  linewidth=2)

    # 添加图例，包含模型名称和50%标记
    all_legend_items = []
    if legend_labels:
        # 添加模型名称到图例
        all_legend_items.extend(legend_labels)

    # 如果有竖线，确保50%标签也在图例中
    if vline_added_to_legend:
        # 图例会自动包含带有label的元素
        pass

    plt.legend(loc='lower right', framealpha=0.9, fancybox=True, shadow=True, fontsize=10,
              bbox_to_anchor=(1.0, 0.0), ncol=1)

    # 设置y轴范围和刻度
    all_values = []
    for model_configs in info_gains.values():
        for config in group_configs:
            if config in model_configs:
                infogain_list = model_configs[config]
                all_values.extend(infogain_list)

    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min

        # 如果所有值相同，设置一个合理的范围
        if y_range == 0:
            if y_max == 0:
                plt.ylim(-0.1, 0.5)
            else:
                plt.ylim(max(0, y_max - 0.5), y_max + 0.5)
        else:
            plt.ylim(max(0, y_min - y_range * 0.1), y_max + y_range * 0.1)

    # 调整布局，为图例留出更多空间
    plt.subplots_adjust(right=0.72)

    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {save_path}")
    plt.show()

    # 打印结果
    print(f"\n{group_name} 结果:")
    for model_name, model_configs in info_gains.items():
        print(f"  {model_name}:")
        for config in group_configs:
            if config in model_configs:
                infogain_list = model_configs[config]
                final_gain = infogain_list[-1] if infogain_list else 0.0
                print(f"    {config}: {len(infogain_list)} turns, final={final_gain:.4f}")




def main():
    """主函数"""
    # 请将此路径修改为您实际的 results 目录
    results_dir = "results"

    # 读取信息增益数据和样本结束步数
    info_gains, sample_end_steps = read_info_gain_from_models(results_dir)
    info_gains.pop("openai/gpt-oss-120b", None)  # 删除oracle基线
    sample_end_steps.pop("openai/gpt-oss-120b", None)  # 删除oracle基线的样本结束步数

    # 添加Strategist模型数据
    strategist_data = [0.1306, 0.2298, 0.326, 0.4124, 0.4642, 0.533, 0.6201, 0.7446, 0.8083, 0.8671, 0.9162, 0.9481, 0.9696, 0.9794, 0.9846, 0.987, 0.9884, 0.9888, 0.9888, 0.9893]
    info_gains["Strategist"] = {"active_text": strategist_data}
    # Strategist不需要竖线，所以不添加到sample_end_steps中
    if not info_gains:
        print("未找到信息增益数据！")
        return

    # 收集所有配置
    all_configs = set()
    for configs in info_gains.values():
        all_configs.update(configs.keys())

    print(f"可用的active配置: {sorted(all_configs)}")

    # 定义配置组
    config_groups = {
        "Active Text": [c for c in all_configs if "text" in c.lower() and "active" in c.lower()],
    }

    # 为每个配置组生成图表（折线图样式）
    for group_name, group_configs in config_groups.items():
        save_path = f"info_gain_across_models.pdf"
        plot_config_group_line_style(info_gains, sample_end_steps, group_name, group_configs, save_path)


if __name__ == "__main__":
    main()