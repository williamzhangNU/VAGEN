"""
Scatter plot visualization for exploration cost vs performance

Generates two scatter plots:
1. Active Text: Exploration cost vs evaluation accuracy
2. Active Vision: Exploration cost vs evaluation accuracy

Usage:
    python plot_cost_performance.py
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


def read_cost_performance_from_models(results_dir: str) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Read env_data.json files and extract exploration cost and evaluation accuracy.

    Returns:
        dict: model_name -> config_name -> (avg_cost, avg_accuracy)
    """
    data = {}

    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

        env_data_path = os.path.join(model_path, "env_data.json")
        if not os.path.exists(env_data_path):
            continue

        try:
            with open(env_data_path, 'r') as f:
                json_data = json.load(f)

            model_name = json_data.get('meta_info', {}).get('model_name', model_dir)
            
            # Skip oracle baselines
            if model_name in ["Qwen3-VL-32B-Instruct", "GLM-4.5V"]:
                continue
            
            exp_summary = json_data.get('exp_summary', {}).get('group_performance', {})
            eval_summary = json_data.get('eval_summary', {}).get('group_performance', {})

            model_configs = {}

            # Extract cost and accuracy from experiment summary and evaluation summary
            for config_name, config_data in exp_summary.items():
                # Only include active configurations
                if "active" not in config_name.lower():
                    continue

                # Get average exploration cost from exp_summary
                avg_cost = config_data.get('avg_action_cost')
                
                # Get evaluation accuracy from eval_summary
                avg_accuracy = None
                if config_name in eval_summary:
                    avg_accuracy = eval_summary[config_name].get('avg_accuracy')

                if avg_cost is not None and avg_accuracy is not None:
                    model_configs[config_name] = (avg_cost, avg_accuracy)

            if model_configs:
                data[model_name] = model_configs
                print(f"Found data for {model_name}: {list(model_configs.keys())}")
                for config, (cost, acc) in model_configs.items():
                    print(f"  {config}: cost={cost:.2f}, accuracy={acc:.4f}")

        except Exception as e:
            print(f"{env_data_path} error: {e}")

    return data


def plot_scatter(data: Dict[str, Dict[str, Tuple[float, float]]], 
                 config_filter: str,
                 title: str,
                 save_path: str) -> None:
    """Create a scatter plot for cost vs performance.
    
    Args:
        data: model_name -> config_name -> (cost, accuracy)
        config_filter: string to filter configurations (e.g., "text" or "vision")
        title: plot title
        save_path: path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Define color palette and marker styles
    colors = ['#4285f4', '#ea4335', '#34a853', '#fbbc05', '#ff6d00', '#795548', '#673ab7', '#9c27b0']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']

    model_index = 0
    
    # Collect all data points for each model
    for model_name, model_configs in sorted(data.items()):
        # Filter configurations based on config_filter
        filtered_configs = {k: v for k, v in model_configs.items() 
                          if config_filter.lower() in k.lower()}
        
        if not filtered_configs:
            continue
        
        # Extract costs and accuracies for this model
        costs = []
        accuracies = []
        
        for config_name, (cost, accuracy) in filtered_configs.items():
            costs.append(cost)
            accuracies.append(accuracy)
        
        if not costs:
            continue
        
        # Get model short name
        model_short = model_name.split('/')[-1] if '/' in model_name else model_name
        
        # Plot scatter points for this model
        color = colors[model_index % len(colors)]
        marker = markers[model_index % len(markers)]
        
        ax.scatter(costs, accuracies, 
                  s=150,  # marker size
                  c=color,
                  marker=marker,
                  alpha=0.7,
                  edgecolors='black',
                  linewidths=1.5,
                  label=model_short)
        
        # Connect points with lines if there are multiple configurations
        if len(costs) > 1:
            # Sort by cost for connecting
            sorted_indices = np.argsort(costs)
            sorted_costs = np.array(costs)[sorted_indices]
            sorted_accs = np.array(accuracies)[sorted_indices]
            
            ax.plot(sorted_costs, sorted_accs,
                   color=color,
                   linestyle='--',
                   linewidth=1.5,
                   alpha=0.3)
        
        model_index += 1

    # Set labels and title
    ax.set_xlabel('Exploration Cost', fontsize=14, fontweight='bold')
    ax.set_ylabel('Evaluation Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Set y-axis limits to show the full range (0 to 1 for accuracy)
    ax.set_ylim([0, 1.05])
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    # Add legend
    ax.legend(loc='best', fontsize=11, framealpha=0.9)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot to {save_path}")
    plt.close()


def main():
    """Main function to generate scatter plots."""
    # Directory containing model results
    results_dir = "/home/zihanhuang/VAGEN/results_arxiv"
    
    # Output directory for plots
    output_dir = "/home/zihanhuang/VAGEN"
    os.makedirs(output_dir, exist_ok=True)

    print("Reading data from models...")
    data = read_cost_performance_from_models(results_dir)

    if not data:
        print("No data found!")
        return

    print("\nGenerating scatter plots...")

    # Plot 1: Active Text
    plot_scatter(
        data=data,
        config_filter="text",
        title="Active Text: Exploration Cost vs Performance",
        save_path=os.path.join(output_dir, "scatter_active_text.png")
    )

    # Plot 2: Active Vision
    plot_scatter(
        data=data,
        config_filter="vision",
        title="Active Vision: Exploration Cost vs Performance",
        save_path=os.path.join(output_dir, "scatter_active_vision.png")
    )

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
