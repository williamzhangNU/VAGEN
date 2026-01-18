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
import sys
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.legend_handler import HandlerBase
from PIL import Image

# Add vagen to path for importing get_exploration_history_stats
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from vagen.env.spatial.Base.tos_base.managers.agent_proxy import get_exploration_history_stats

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


class ImageHandler(HandlerBase):
    """Custom legend handler for images"""
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        if isinstance(orig_handle, OffsetImage):
            bb = AnnotationBbox(orig_handle, (width/2, height/2), 
                              frameon=False, 
                              xycoords=trans,
                              box_alignment=(0.5, 0.5))
            return [bb]
        return []


def preprocess_icon(img_path: str, target_size: int = 300, grayscale: bool = False) -> Image.Image:
    """Load and preprocess icon to uniform size.
    
    Args:
        img_path: Path to the icon image
        target_size: Target size for the square icon (default: 300x300)
        grayscale: Whether to convert to grayscale (for scout icons)
    
    Returns:
        Preprocessed PIL Image
    """
    img = Image.open(img_path)
    
    # Convert to RGBA if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Convert to grayscale if requested
    if grayscale:
        # Convert RGB channels to grayscale while preserving alpha
        rgb = img.convert('RGB').convert('L').convert('RGB')
        # Merge back with original alpha channel and reduce opacity
        alpha = img.split()[3]
        # Make it lighter by reducing alpha to 50%
        alpha = alpha.point(lambda p: int(p * 0.5))
        img = Image.merge('RGBA', (*rgb.split(), alpha))
    
    # Resize to target size (square)
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    return img

def read_cost_performance_from_models(results_dir: str) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Read env_data.json files and extract exploration cost and evaluation accuracy.
    
    Args:
        results_dir: Directory containing model results
    
    Returns:
        dict: model_name -> config_name -> (avg_cost, avg_accuracy)
    """
    data = {}
    scout_cost = get_exploration_history_stats('/home/zihanhuang/VAGEN/data-3room/tos_dataset_0109_3room_100runs', 'scout').get('avg_action_cost', 0.0)
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
            
            # Add Vision Scout data if available
            if model_name not in ["Qwen3-VL-32B-Instruct", "GLM-4.5V"]:
                # Get evaluation accuracy for Vision Scout from eval_summary
                scout_config = "vision_passive_think_scout"
                if scout_config in eval_summary:
                    scout_accuracy = eval_summary[scout_config].get('avg_accuracy')
                    if scout_accuracy is not None:
                        # Calculate scout cost from all samples in env_data
                        model_configs["Vision Scout"] = (scout_cost, scout_accuracy)
                        print(f"  {model_name}: Added Vision Scout")
                        print(f"    cost={scout_cost:.2f}, accuracy={scout_accuracy:.4f}")
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
                 save_path: str,
                 include_scout: bool = False) -> None:
    """Create a scatter plot for cost vs performance.
    
    Args:
        data: model_name -> config_name -> (cost, accuracy)
        config_filter: string to filter configurations (e.g., "text" or "vision")
        title: plot title
        save_path: path to save the figure
        include_scout: whether to include Vision Scout data
    """
    fig, ax = plt.subplots(figsize=(4, 5))

    # Define icon paths - map model keywords to icon files
    icon_dir = "./icon"
    model_icon_map = {
        'claude': 'claude.png',
        'gemini': 'gemini.png',
        'glm': 'glm.png',
        'gpt': 'gpt.png',
        'qwen': 'qwen.png'
    }
    
    # Define color palette
    colors = ['#4285f4', '#ea4335', '#34a853', '#fbbc05', '#ff6d00', '#795548', '#673ab7', '#9c27b0']

    model_index = 0
    
    # Track all costs for x-axis range calculation
    all_costs = []
    
    # Collect all data points for each model
    for model_name, model_configs in sorted(data.items()):
        # Filter configurations based on config_filter
        filtered_configs = {k: v for k, v in model_configs.items() 
                          if config_filter.lower() in k.lower()}
        
        # Include Vision Scout if requested
        if include_scout and "Vision Scout" in model_configs:
            filtered_configs["Vision Scout"] = model_configs["Vision Scout"]
        
        if not filtered_configs:
            continue
        
        # Separate scout and active configurations
        scout_data = []
        active_data = []
        
        for config_name, (cost, accuracy) in filtered_configs.items():
            if config_name == "Vision Scout":
                scout_data.append((cost, accuracy))
            else:
                active_data.append((cost, accuracy))
        
        if not scout_data and not active_data:
            continue
        
        # Get model short name
        model_short = model_name.split('/')[-1] if '/' in model_name else model_name
        
        # Get color for this model
        color = colors[model_index % len(colors)]
        
        # Find matching icon based on model name
        icon_path = None
        model_name_lower = model_name.lower()
        for keyword, icon_file in model_icon_map.items():
            if keyword in model_name_lower:
                icon_path = os.path.join(icon_dir, icon_file)
                break
        
        # Plot active configurations
        if active_data:
            active_costs = [c for c, a in active_data]
            active_accs = [a for c, a in active_data]
            
            # Track costs for x-axis range
            all_costs.extend(active_costs)
            
            # Use image as marker
            if icon_path and os.path.exists(icon_path):
                img = preprocess_icon(icon_path)
                for cost, acc in zip(active_costs, active_accs):
                    imagebox = OffsetImage(img, zoom=0.03)
                    ab = AnnotationBbox(imagebox, (cost, acc),
                                      frameon=False,
                                      pad=0)
                    ax.add_artist(ab)
            else:
                # Fallback to regular scatter if image not found
                ax.scatter(active_costs, active_accs, 
                          s=50,
                          c=color,
                          marker='o',
                          alpha=0.7,
                          edgecolors='black',
                          linewidths=0.8)
            
            # Add to legend (create a dummy scatter for legend entry)
            ax.scatter([], [], s=50, c=color, marker='o', 
                      edgecolors='black', linewidths=0.8,
                      label=model_short)
            
            # Connect active points with lines if multiple
            if len(active_costs) > 1:
                sorted_indices = np.argsort(active_costs)
                sorted_costs = np.array(active_costs)[sorted_indices]
                sorted_accs = np.array(active_accs)[sorted_indices]
                
                ax.plot(sorted_costs, sorted_accs,
                       color=color,
                       linestyle='--',
                       linewidth=0.8,
                       alpha=0.3)
        
        # Plot scout with grayscale icon
        if scout_data:
            scout_costs = [c for c, a in scout_data]
            scout_accs = [a for c, a in scout_data]
            
            # Track costs for x-axis range
            all_costs.extend(scout_costs)
            
            # Use grayscale image as marker for scout
            if icon_path and os.path.exists(icon_path):
                img_gray = preprocess_icon(icon_path, grayscale=True)
                for cost, acc in zip(scout_costs, scout_accs):
                    imagebox = OffsetImage(img_gray, zoom=0.03)
                    ab = AnnotationBbox(imagebox, (cost, acc),
                                      frameon=False,
                                      pad=0)
                    ax.add_artist(ab)
                
                # Add to legend (create a dummy scatter for legend entry)
                ax.scatter([], [], s=100, c='gray', marker='*', 
                          edgecolors='black', linewidths=1,
                          label=f"{model_short} (Scout)")
            else:
                # Fallback to star marker if image not found
                ax.scatter(scout_costs, scout_accs, 
                          s=100,  # larger for visibility
                          c='gray',
                          marker='*',  # star marker for scout
                          alpha=0.9,
                          edgecolors='black',
                          linewidths=1,
                          label=f"{model_short} (Scout)")
        
        model_index += 1

    # Set labels and title
    ax.set_xlabel('Exploration Cost', fontsize=10, fontweight='bold')
    ax.set_ylabel('Evaluation Accuracy', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Set axis limits
    ax.set_ylim([0, 1.05])
    
    # Set x-axis range to fit all data with small margins
    if all_costs:
        min_cost = min(all_costs)
        max_cost = max(all_costs)
        margin = (max_cost - min_cost) * 0.1  # 10% margin
        ax.set_xlim([max(0, min_cost - margin), max_cost + margin])
    
    # Let x-axis auto-scale to fit data (removed fixed MultipleLocator)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    # Add custom legend with icons
    handles, labels = ax.get_legend_handles_labels()
    
    # Create custom legend handles with images based on model name
    from matplotlib.patches import Rectangle
    custom_handles = []
    for i, (handle, label) in enumerate(zip(handles, labels)):
        # Find matching icon based on label (model name)
        icon_path = None
        label_lower = label.lower()
        for keyword, icon_file in model_icon_map.items():
            if keyword in label_lower:
                icon_path = os.path.join(icon_dir, icon_file)
                break
        
        if icon_path and os.path.exists(icon_path):
            if "(Scout)" in label:
                # For scout models, use grayscale icon
                img = preprocess_icon(icon_path, target_size=160, grayscale=True)
                imagebox = OffsetImage(img, zoom=0.05)
                custom_handles.append(imagebox)
            else:
                # For active models, use color icon
                img = preprocess_icon(icon_path, target_size=160)
                imagebox = OffsetImage(img, zoom=0.05)
                custom_handles.append(imagebox)
        else:
            # For scout or fallback, use original handle
            custom_handles.append(handle)
    
    ax.legend(custom_handles, labels, loc='best', fontsize=7, framealpha=0.9,
             handler_map={OffsetImage: ImageHandler()})

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
        save_path=os.path.join(output_dir, "scatter_active_vision.png"),
        include_scout=True
    )

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
