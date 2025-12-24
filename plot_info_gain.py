"""
Simplified information gain visualization script

Generates two plots:
1. Information gain comparison for Active Text configurations
2. Information gain comparison for Active Vision configurations

Usage:
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
    """Read env_data.json files under different model directories and extract
    information gain values and sample end steps (active configurations only).

    Returns:
        tuple: (info_gains, sample_end_steps)
        - info_gains: model_name -> config_name -> list of information gains
        - sample_end_steps: model_name -> config_name -> list of end step numbers
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

            # Extract information gain from aggregated experiment summary
            for config_name, config_data in exp_summary.items():
                # Only include configurations that are active
                if "active" not in config_name.lower():
                    continue

                infogain_per_turn = config_data.get('infogain_per_turn')
                if infogain_per_turn is not None and isinstance(infogain_per_turn, list) and len(infogain_per_turn) > 0:
                    model_configs[config_name] = infogain_per_turn
                else:
                    # Fall back to using avg_final_information_gain as a single value
                    final_info_gain = config_data.get('avg_final_information_gain')
                    if final_info_gain is not None:
                        model_configs[config_name] = [final_info_gain]  # convert to list format

            # For each config, collect per-sample end steps from env_turn_logs
            for config_name in model_configs.keys():
                end_steps = []
                for sample_name, sample_data in samples.items():
                    if config_name in sample_data:
                        config_sample_data = sample_data[config_name]
                        if isinstance(config_sample_data, dict) and 'env_turn_logs' in config_sample_data:
                            turn_logs = config_sample_data['env_turn_logs']
                            if turn_logs and len(turn_logs) > 0:
                                # Use the last turn's 'turn_number' as the end step
                                last_turn_number = turn_logs[-1].get('turn_number', 0)
                                end_steps.append(last_turn_number)

                if end_steps:
                    model_end_steps[config_name] = end_steps

            if model_configs:
                info_gains[model_name] = model_configs
                sample_end_steps[model_name] = model_end_steps
                print(f"Found information gain for {model_name}: {list(model_configs.keys())}")
                print(f"  Sample end steps: {[(config, len(steps)) for config, steps in model_end_steps.items()]}")

        except Exception as e:
            print(f"{env_data_path} error: {e}")

    return info_gains, sample_end_steps


def plot_config_group_line_style(info_gains: Dict[str, Dict[str, List[float]]],
                                sample_end_steps: Dict[str, Dict[str, List[int]]],
                                group_name: str,
                                group_configs: list,
                                save_path: str) -> None:
    """Create a line-style plot for a group of configurations (visual style is tuned).
    """
    if not group_configs:
        return

    # Prepare data - create a separate line for each configuration
    plt.figure(figsize=(8, 4))

    # Define color palette and marker styles
    colors = ['#4285f4', '#ea4335', '#34a853', '#fbbc05', '#ff6d00', '#795548', '#673ab7']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    linestyles = ['-']

    # Collect data across models to determine the maximum number of turns
    max_turns = 0

    # Collect data for all models in this configuration group
    model_data = {}
    for model_name, model_configs in info_gains.items():
        for config in group_configs:
            if config in model_configs:
                infogain_list = model_configs[config]
                # Create a unique key for each model-config combination
                model_config_key = f"{model_name}_{config}"
                model_data[model_config_key] = {
                    'model_name': model_name,
                    'config': config,
                    'data': infogain_list
                }
                max_turns = max(max_turns, len(infogain_list))

    if max_turns == 0:
        return

    # x-axis positions represent step numbers (starting from 1)
    x_positions = range(1, max_turns + 1)

    # Plot a line for each model-config combination
    line_index = 0
    legend_labels = []
    model_colors = {} 

    for model_config_key, data_info in model_data.items():
        model_name = data_info['model_name']
        config = data_info['config']
        infogain_list = data_info['data']

        if infogain_list:
            # Create legend label showing only the model short name
            model_short = model_name.split('/')[-1] if '/' in model_name else model_name

            # Use a distinct color per plotted line
            line_color = colors[line_index % len(colors)]

            # Prepare x and y data limited to the available length
            x_data = x_positions[:len(infogain_list)]
            y_data = infogain_list

            # Plot the full line
            plt.plot(x_data, y_data,
                    color=line_color,
                    linestyle=linestyles[line_index % len(linestyles)],
                    linewidth=2.5,
                    alpha=0.8,
                    label=model_short)

            # Show markers only on odd steps to reduce clutter
            for i, (x, y) in enumerate(zip(x_data, y_data)):
                if x % 2 == 1:  # odd step numbers
                    plt.plot(x, y,
                            marker=markers[line_index % len(markers)],
                            color=line_color,
                            markersize=5,
                            markerfacecolor='white',
                            markeredgewidth=2)

            legend_labels.append(model_short)

            # Store color for this model for later reference
            model_colors[model_name] = line_color

            line_index += 1

    # Customize plot appearance
    plt.title(f'Accumulated Information Gain', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Steps', fontsize=14, fontweight='bold')
    plt.ylabel('Information Gain', fontsize=14, fontweight='bold')

    # Set x-axis ticks - show every other tick (1,3,5,...)
    x_ticks = [x for x in x_positions if x % 2 == 1]  # show 1,3,5,7...
    plt.xticks(x_ticks, fontsize=14)
    plt.yticks(fontsize=14)

    # Add a light grid
    plt.grid(True, alpha=0.3, linestyle='--')

    # Draw vertical lines indicating sample end steps (percentile-based)
    vline_positions = {}  # track counts per position to offset overlapping lines
    vline_added_to_legend = False  # ensure the line label is added to legend only once

    for model_name, model_end_steps in sample_end_steps.items():
        for config in group_configs:
            if config in model_end_steps:
                end_steps_list = model_end_steps[config]
                if end_steps_list:
                    # Compute the 80th percentile of end steps
                    end_steps_sorted = sorted(end_steps_list)
                    median_step = np.percentile(end_steps_sorted, 80)

                    # Compute a small offset to avoid overlapping vertical lines
                    if median_step in vline_positions:
                        vline_positions[median_step] += 1
                        offset = (vline_positions[median_step] - 1) * 0.1 - 0.05
                    else:
                        vline_positions[median_step] = 1
                        offset = 0

                    # Use the model's line color for the vertical marker
                    vline_color = model_colors.get(model_name, 'gray')

                    # Add the first vertical line to the legend as a gray dashed entry
                    if not vline_added_to_legend:
                        plt.axvline(x=median_step + offset,
                                  color=vline_color,
                                  linestyle='--',
                                  alpha=0.7,
                                  linewidth=2)
                        # Create a separate gray legend entry for sample-end markers
                        plt.plot([], [], color='gray', linestyle='--',
                               linewidth=2, label='80% Samples Ends')
                        vline_added_to_legend = True
                    else:
                        plt.axvline(x=median_step + offset,
                                  color=vline_color,
                                  linestyle='--',
                                  alpha=0.7,
                                  linewidth=2)

    # Add legend including model names and the sample-end marker
    all_legend_items = []
    if legend_labels:
        all_legend_items.extend(legend_labels)

    if vline_added_to_legend:
        pass

    # Compact the legend inside the plot (lower right) for a tighter layout
    plt.legend(loc='lower right', framealpha=0.85, fancybox=True, shadow=True,
               fontsize=8, markerscale=0.7, handlelength=1.2, handletextpad=0.4,
               borderpad=0.3, bbox_to_anchor=(0.98, 0.02), ncol=1)

    # Set y-axis limits and ticks
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

        # If all values are equal, set a sensible range
        if y_range == 0:
            if y_max == 0:
                plt.ylim(-0.1, 0.5)
            else:
                plt.ylim(max(0, y_max - 0.5), y_max + 0.5)
        else:
            plt.ylim(max(0, y_min - y_range * 0.1), y_max + y_range * 0.1)

    # Adjust layout and tighten right margin
    plt.subplots_adjust(right=0.90)

    # Save plot to file
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"save to: {save_path}")
    plt.show()

    # Print summary results
    print(f"\n{group_name} result:")
    for model_name, model_configs in info_gains.items():
        print(f"  {model_name}:")
        for config in group_configs:
            if config in model_configs:
                infogain_list = model_configs[config]
                final_gain = infogain_list[-1] if infogain_list else 0.0
                print(f"    {config}: {len(infogain_list)} turns, final={final_gain:.4f}")




def main():
    """Main entrypoint"""
    # 请将此路径修改为您实际的 results 目录
    results_dir = "/home/zihanhuang/VAGEN/results_arxiv"

    # Read information gain data and sample end steps
    info_gains, sample_end_steps = read_info_gain_from_models(results_dir)
    info_gains.pop("Qwen3-VL-32B-Instruct", None)  # 删除oracle基线
    info_gains.pop("GLM-4.5V", None)  # 删除GLM-4.5V基线
    sample_end_steps.pop("Qwen3-VL-32B-Instruct", None)  # 删除oracle基线的样本结束步数
    sample_end_steps.pop("GLM-4.5V", None)  # 删除GLM-4.5V基线的样本结束步数

    # Add Strategist model data (manually provided)
    strategist_data = [0.1045, 0.1903, 0.2669, 0.3584, 0.4007, 0.4582, 0.5107, 0.602, 0.6484, 0.6855, 0.7253, 0.7652, 0.8244, 0.8771, 0.9202, 0.9463, 0.9642, 0.976, 0.9821, 0.985, 0.9882]
    info_gains["Strategist"] = {"active_text": strategist_data}
    # Strategist不需要竖线，所以不添加到sample_end_steps中
    if not info_gains:
        print("未找到信息增益数据！")
        return

    # Collect all configurations present in the data
    all_configs = set()
    for configs in info_gains.values():
        all_configs.update(configs.keys())

    print(f"可用的active配置: {sorted(all_configs)}")

    # Define configuration groups
    config_groups = {
        "Active Text": [c for c in all_configs if "text" in c.lower() and "active" in c.lower()],
    }

    # Generate plots (line style) for each configuration group
    for group_name, group_configs in config_groups.items():
        save_path = f"info_gain_across_models.pdf"
        plot_config_group_line_style(info_gains, sample_end_steps, group_name, group_configs, save_path)


if __name__ == "__main__":
    main()