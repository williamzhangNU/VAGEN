"""
Scatter plot visualization for exploration cost vs performance

Generates two scatter plots:
1. Active Text: Exploration cost vs evaluation accuracy
2. Active Vision: Exploration cost vs evaluation accuracy
3. Active Combined (Text & Vision): Exploration cost vs evaluation accuracy

Usage:
    python analysis/plot_cost_performance.py --results_dir /path/to/results --output_dir /path/to/output --data_dir /path/to/data
"""

import os
import json
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
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
            bb = AnnotationBbox(orig_handle, (width / 2, height / 2),
                                frameon=False,
                                xycoords=trans,
                                box_alignment=(0.5, 0.5))
            return [bb]
        return []


def preprocess_icon(img_path: str, target_size: int = 256, grayscale: bool = False) -> Image.Image:
    """Load, crop transparent borders (to fix alignment), and resize icon."""
    img = Image.open(img_path)

    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # Auto-crop to content: Removes transparent whitespace so the icon centers correctly
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    if grayscale:
        rgb = img.convert('RGB').convert('L').convert('RGB')
        alpha = img.split()[3]
        alpha = alpha.point(lambda p: int(p * 0.5))
        img = Image.merge('RGBA', (*rgb.split(), alpha))

    # Uniform resize
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return img


def read_cost_performance_from_models(
    results_dir: str,
    data_dir: Optional[str] = None
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Read env_data.json files and extract exploration cost and evaluation accuracy."""
    data: Dict[str, Dict[str, Tuple[float, float]]] = {}
    scout_cost = 0.0
    strategist_cost = 0.0

    MODEL_MAPPING = {
        "claude-sonnet-4-5": "Claude-4.5-Sonnet",
        "gemini-3-pro-preview": "Gemini-3-Pro",
        "qwen/qwen3-vl-235b-a22b-thinking": "Qwen3-VL",
        "glm-4.6v": "GLM-4.6V",
        "gpt-5.2": "GPT-5.2",
    }

    # Exclude InternVL (and variants) from all plots
    EXCLUDE_MODEL_KEYWORDS = ["internvl"]

    if data_dir and os.path.exists(data_dir):
        try:
            print(f"Calculating baseline costs from {data_dir}...")
            scout_stats = get_exploration_history_stats(data_dir, 'scout')
            scout_cost = scout_stats.get('avg_action_cost', 0.0)

            strategist_stats = get_exploration_history_stats(data_dir, 'strategist')
            strategist_cost = strategist_stats.get('avg_action_cost', 0.0)
            print(f"Baseline costs: Scout={scout_cost:.2f}, Strategist={strategist_cost:.2f}")
        except Exception as e:
            print(f"Warning: Failed to load baseline costs: {e}")

    if not os.path.exists(results_dir):
        print(f"Warning: results_dir does not exist: {results_dir}")
        return data

    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

        # Exclude by folder name too (robust)
        if any(k in model_dir.lower() for k in EXCLUDE_MODEL_KEYWORDS):
            continue

        env_data_path = os.path.join(model_path, "env_data.json")
        if not os.path.exists(env_data_path):
            continue

        try:
            with open(env_data_path, 'r') as f:
                json_data = json.load(f)

            raw_name = json_data.get('meta_info', {}).get('model_name', model_dir)

            # Exclude by raw name (robust)
            if any(k in str(raw_name).lower() for k in EXCLUDE_MODEL_KEYWORDS):
                continue

            model_name = MODEL_MAPPING.get(raw_name, raw_name)

            # Exclude by mapped name (robust)
            if any(k in str(model_name).lower() for k in EXCLUDE_MODEL_KEYWORDS):
                continue

            exp_summary = json_data.get('exp_summary', {}).get('group_performance', {})
            eval_summary = json_data.get('eval_summary', {}).get('group_performance', {})

            model_configs: Dict[str, Tuple[float, float]] = {}

            # Process active configurations
            for config_name, config_data in exp_summary.items():
                if "active" not in config_name.lower():
                    continue

                avg_cost = config_data.get('avg_action_cost')
                avg_accuracy = None
                if config_name in eval_summary:
                    avg_accuracy = eval_summary[config_name].get('avg_accuracy')

                if avg_cost is not None and avg_accuracy is not None:
                    model_configs[config_name] = (avg_cost, avg_accuracy)

            # Add baselines
            if scout_cost > 0:
                scout_keys = [k for k in eval_summary.keys()
                              if "scout" in k.lower() and "vision" in k.lower()]
                for key in scout_keys:
                    acc = eval_summary[key].get('avg_accuracy')
                    if acc is not None:
                        model_configs["Vision Scout"] = (scout_cost, acc)
                        break

            if strategist_cost > 0:
                strat_keys = [k for k in eval_summary.keys()
                              if "strategist" in k.lower() and "text" in k.lower()]
                for key in strat_keys:
                    acc = eval_summary[key].get('avg_accuracy')
                    if acc is not None:
                        model_configs["Text Strategist"] = (strategist_cost, acc)
                        break

            if model_configs:
                data[str(model_name)] = model_configs

        except Exception as e:
            print(f"{env_data_path} error: {e}")

    return data


def solve_overlaps(points: List[Tuple[float, float]], x_range: float, y_range: float) -> List[Tuple[float, float]]:
    """Simple repulsion algorithm to separate overlapping points."""
    n = len(points)
    if n <= 1:
        return points

    norm_points = np.array([
        [x / x_range if x_range > 0 else 0, y / y_range if y_range > 0 else 0]
        for x, y in points
    ])

    current_pos = norm_points.copy()
    for _ in range(50):
        forces = np.zeros_like(current_pos)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                diff = current_pos[i] - current_pos[j]
                dist = np.linalg.norm(diff)
                threshold = 0.1
                if dist < threshold:
                    force = diff / (dist + 1e-6) * (threshold - dist)
                    forces[i] += force
        current_pos += forces * 0.1

    return [(current_pos[i, 0] * x_range, current_pos[i, 1] * y_range) for i in range(n)]


def plot_scatter(
    data: Dict[str, Dict[str, Tuple[float, float]]],
    config_filter: str,
    title: str,
    save_path: str,
    baseline_key: Optional[str] = None,
    grayscale_keyword: Optional[str] = None,
    connect_points: bool = True,
) -> None:
    """Create a scatter plot for cost vs performance."""
    fig, ax = plt.subplots(figsize=(6.5, 5))

    icon_dir = os.path.join(os.path.dirname(__file__), "icon")
    model_icon_map = {
        'claude': 'claude.png',
        'gemini': 'gemini.png',
        'glm': 'glm.png',
        'gpt': 'gpt.png',
        'qwen': 'qwen.png'
    }

    ICON_ZOOM_DEFAULT = 0.14
    ICON_ZOOM_RULES = [
        ("claude-4.5-sonnet", 0.14),
        ("gemini 3 pro", 0.14),
        ("qwen3 vl", 0.14),
        ("glm-4.6v", 0.14),
        ("gpt-5.2", 0.14),
    ]

    def get_icon_zoom(name: str) -> float:
        s = name.lower()
        for key, z in ICON_ZOOM_RULES:
            if key.lower() in s:
                return z
        return ICON_ZOOM_DEFAULT

    colors = ['#4285f4', '#ea4335', '#34a853', '#fbbc05', '#ff6d00', '#795548', '#673ab7', '#9c27b0']
    model_index = 0
    all_points = []

    for model_name, model_configs in sorted(data.items()):
        filtered_configs = {k: v for k, v in model_configs.items() if config_filter.lower() in k.lower()}
        if baseline_key and baseline_key in model_configs:
            filtered_configs[baseline_key] = model_configs[baseline_key]

        if not filtered_configs:
            continue

        color = colors[model_index % len(colors)]
        icon_path = next((os.path.join(icon_dir, f) for k, f in model_icon_map.items() if k in model_name.lower()), None)

        model_short = model_name.split('/')[-1] if '/' in model_name else model_name
        ax.scatter([], [], s=100, c=color, marker='o', edgecolors='black', linewidths=1.5, label=model_short)

        active_points = []
        for config_name, (cost, accuracy) in filtered_configs.items():
            is_baseline = (config_name == baseline_key)
            is_gray = is_baseline or (grayscale_keyword and grayscale_keyword.lower() in config_name.lower())

            all_points.append({
                'x': cost,
                'y': accuracy,
                'is_baseline': is_baseline,
                'is_gray': is_gray,
                'icon_path': icon_path,
                'icon_zoom': get_icon_zoom(model_name),
                'color': color
            })
            if not is_baseline:
                active_points.append((cost, accuracy))

        # Draw dashed line connecting points (optional)
        if connect_points and len(active_points) > 1:
            active_points.sort()
            xs, ys = zip(*active_points)
            ax.plot(xs, ys, color=color, linestyle='--', linewidth=1.5, alpha=0.4)

        model_index += 1

    if not all_points:
        print(f"No points to plot for {config_filter}.")
        plt.close()
        return

    # Plot points with repulsion
    xs = [p['x'] for p in all_points]
    x_range = max(xs) - min(xs) if xs and max(xs) > min(xs) else 1.0
    adjusted_xy = solve_overlaps([(p['x'], p['y']) for p in all_points], x_range, 1.0)

    for i, p in enumerate(all_points):
        orig_x, orig_y = p['x'], p['y']
        new_x, new_y = adjusted_xy[i]

        if p['icon_path'] and os.path.exists(p['icon_path']):
            img = preprocess_icon(p['icon_path'], grayscale=p.get('is_gray', False))
            imagebox = OffsetImage(img, zoom=p['icon_zoom'] * 0.7)
            ab = AnnotationBbox(
                imagebox,
                (orig_x, orig_y),
                xybox=(new_x, new_y),
                boxcoords="data",
                frameon=False,
                pad=0,
                arrowprops=dict(arrowstyle="-", color="black", alpha=0.3, lw=0.5)
            )
            ax.add_artist(ab)
        else:
            marker = '*' if p['is_baseline'] else 'o'
            ax.scatter(orig_x, orig_y, s=150, c=p['color'], marker=marker, edgecolors='black', linewidths=1.5, zorder=2)

    # Style
    ax.set_title(title, fontsize=16, fontweight='bold', pad=40)
    ax.set_xlabel('Exploration Cost', fontsize=14, fontweight='bold')
    ax.set_ylabel('Evaluation Accuracy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)
    ax.set_ylim([0, 1.05])

    if xs:
        margin = (max(xs) - min(xs)) * 0.15
        ax.set_xlim([max(0, min(xs) - margin), max(xs) + margin])

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    custom_handles = []
    for handle, label in zip(handles, labels):
        icon_path = next((os.path.join(icon_dir, f) for k, f in model_icon_map.items() if k in label.lower()), None)
        if icon_path and os.path.exists(icon_path):
            img = preprocess_icon(icon_path)
            imagebox = OffsetImage(img, zoom=get_icon_zoom(label) * 0.35)
            custom_handles.append(imagebox)
        else:
            custom_handles.append(handle)

    ax.legend(
        custom_handles, labels,
        loc='lower center', bbox_to_anchor=(0.5, 1.02),
        fontsize=10, framealpha=0.9, handler_map={OffsetImage: ImageHandler()},
        ncol=len(custom_handles), borderpad=0.25, labelspacing=0.25,
        handletextpad=0.35, columnspacing=0.6, borderaxespad=0.15
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate cost-performance scatter plots.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Reading data from {args.results_dir}...")
    data = read_cost_performance_from_models(args.results_dir, args.data_dir)

    if not data:
        print("No data found!")
        return

    print("\nGenerating scatter plots...")

    # Text-only: keep dashed connections (default)
    plot_scatter(
        data,
        "text",
        "Active Text: Exploration Cost vs Performance",
        os.path.join(args.output_dir, "scatter_active_text.png"),
        baseline_key="Text Strategist",
        connect_points=True
    )

    # Vision-only: keep dashed connections (default)
    plot_scatter(
        data,
        "vision",
        "Active Vision: Exploration Cost vs Performance",
        os.path.join(args.output_dir, "scatter_active_vision.png"),
        baseline_key="Vision Scout",
        connect_points=True
    )

    # Combined: remove dashed line between text and vision points/icons
    plot_scatter(
        data,
        "active",
        "Active Text & Vision: Exploration Cost vs Performance",
        os.path.join(args.output_dir, "scatter_active_combined.png"),
        baseline_key=None,
        grayscale_keyword="text",
        connect_points=False  # <-- key change: no dashed line in combined plot
    )

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
