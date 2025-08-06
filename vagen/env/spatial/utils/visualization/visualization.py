# visualization.py
import json
import os
import re
from pathlib import Path
from html import escape
from itertools import zip_longest
from typing import List, Dict, Optional
import copy

from .html_templates import HTML_TEMPLATE, CSS_STYLES, JAVASCRIPT_CODE



class VisualizationHelper:
    """Helper class for data processing and HTML generation"""
    
    @staticmethod
    def dict_to_html(d: Dict) -> str:
        """Convert dictionary to HTML format with better styling"""
        if not d:
            return "<div class='empty-dict'>(none)</div>"
        
        html = "<div class='dict-container'>"
        for k, v in d.items():
            if isinstance(v, (int, float)):
                # Format numbers nicely
                if isinstance(v, float):
                    formatted_v = f"{v:.3f}" if v != int(v) else str(int(v))
                else:
                    formatted_v = str(v)
                html += f"<div class='dict-item'><span class='dict-key'>{escape(str(k))}:</span> <span class='dict-value number'>{formatted_v}</span></div>"
            elif isinstance(v, bool):
                # Color-code booleans
                color_class = "true" if v else "false"
                html += f"<div class='dict-item'><span class='dict-key'>{escape(str(k))}:</span> <span class='dict-value {color_class}'>{str(v)}</span></div>"
            elif isinstance(v, dict):
                # Handle nested dictionaries
                nested_html = VisualizationHelper.dict_to_html(v)
                html += f"<div class='dict-item nested'><span class='dict-key'>{escape(str(k))}:</span> <div class='dict-value nested-dict'>{nested_html}</div></div>"
            else:
                # String values with consistent styling
                html += f"<div class='dict-item'><span class='dict-key'>{escape(str(k))}:</span> <span class='dict-value string'>{escape(str(v))}</span></div>"
        html += "</div>"
        return html
    
    @staticmethod
    def extract_think_and_answer(text: str) -> tuple[str, str]:
        """Extract think and answer content from text using regex patterns"""
        think_pattern = r'<think>(.*?)</think>'
        answer_pattern = r'<answer>(.*?)</answer>'
        
        think_match = re.search(think_pattern, text, re.DOTALL)
        answer_match = re.search(answer_pattern, text, re.DOTALL)
        
        think_content = think_match.group(1).strip() if think_match else text
        answer_content = answer_match.group(1).strip() if answer_match else text
        
        return think_content, answer_content








class HTMLGenerator:
    """Handles HTML generation for the visualization"""
    
    def __init__(self, data: Dict, output_html: str, save_images: bool = True):
        self.data = data
        self.output_html = output_html
        self.save_images = save_images
        self.out_dir = os.path.dirname(output_html)
        self.base = Path(output_html).stem
        
        # Extract data
        self.meta = data.get("meta_info", {})
        self.config_groups = data["config_groups"]
        self.total_groups = len(self.config_groups)
        self.overall = data.get("overall_performance", {})
        
        # Extract summary data
        self.exp_summary = data.get("exp_summary", {})
        self.eval_summary = data.get("eval_summary", {})
        self.cogmap_summary = data.get("cogmap_summary", {})
        
        # Calculate statistics
        self.samples_per_group = {
            gname: len(group["env_data"])
            for gname, group in self.config_groups.items()
        }
        self.total_samples = sum(self.samples_per_group.values())
        self.total_pages = 1 + self.total_samples  # page 0 = TOC
        
        # Build flat list
        self.flat = []
        for gname, group in self.config_groups.items():
            for sidx, entry in enumerate(group["env_data"]):
                self.flat.append((gname, sidx, entry))

    def generate_summary_section(self, f) -> None:
        """Generate summary section with overall and group performance"""
        f.write("<div class='summary-section'>\n")
        f.write("<h3>📊 Overall Performance Summary</h3>\n")
        
        # Overall exploration efficiency
        if self.exp_summary.get("overall_performance"):
            exp_overall = self.exp_summary["overall_performance"]
            f.write("<div class='summary-card exploration'>\n")
            f.write("<h4>🔍 Exploration Summary</h4>\n")
            f.write(VisualizationHelper.dict_to_html(exp_overall))
            f.write("</div>\n")
        
        # Overall evaluation performance
        if self.eval_summary.get("overall_performance"):
            eval_overall = self.eval_summary["overall_performance"]
            f.write("<div class='summary-card evaluation'>\n")
            f.write("<h4>✅ Evaluation Summary</h4>\n")
            f.write(VisualizationHelper.dict_to_html(eval_overall))
            f.write("</div>\n")
        
        f.write("</div>\n")

    def generate_config_summaries(self, f) -> None:
        """Generate summaries for each config group"""
        f.write("<div class='config-summaries'>\n")
        f.write("<h3>📋 Configuration Summaries</h3>\n")
        
        for gname, group in self.config_groups.items():
            f.write(f"<div class='config-summary'>\n")
            f.write(f"<h4>⚙️ {escape(gname)}</h4>\n")
            f.write(f"<div class='config-stats'>\n")
            f.write(f"<div class='stat-item'>📊 Samples: {len(group['env_data'])}</div>\n")
            
            # Group exploration performance
            if self.exp_summary.get("group_performance", {}).get(gname):
                exp_group = self.exp_summary["group_performance"][gname]
                f.write("<div class='group-metrics'>")
                f.write("<strong>Exploration:</strong>")
                f.write(VisualizationHelper.dict_to_html(exp_group))
                f.write("</div>\n")
            
            # Group evaluation performance
            if self.eval_summary.get("group_performance", {}).get(gname):
                eval_group = self.eval_summary["group_performance"][gname]
                f.write("<div class='group-metrics'>")
                f.write("<strong>Evaluation:</strong>")
                f.write(VisualizationHelper.dict_to_html(eval_group))
                f.write("</div>\n")

            # Group cognitive map performance
            if self.cogmap_summary.get("group_performance", {}).get(gname):
                cogmap_group = self.cogmap_summary["group_performance"][gname]
                f.write("<div class='group-metrics'>")
                f.write("<strong>Cognitive Map:</strong>")
                f.write(VisualizationHelper.dict_to_html(cogmap_group))
                f.write("</div>\n")
            
            f.write("</div>\n")
            f.write("</div>\n")
        
        f.write("</div>\n")

    def generate_toc_page(self, f) -> None:
        """Generate table of contents page with summaries"""
        f.write("<section class='sample-page active' id='page0'>\n")
        f.write("<h2>📋 Dashboard Overview</h2>\n")
        
        # Add summary sections
        self.generate_summary_section(f)
        self.generate_config_summaries(f)
        
        f.write("<h3>📖 Sample Navigation</h3>\n")
        f.write("<ul>\n")
        running_page = 1
        
        for gname, group in self.config_groups.items():
            f.write(f"<li><strong>{escape(gname)}</strong>\n  <ul>\n")
            for idx in range(len(group["env_data"])):
                label = f"Sample {idx+1}"
                f.write(
                    f"    <li>"
                    f"<a href='#' onclick=\"showPage({running_page}, {self.total_pages});return false;\">"
                    f"{label}</a></li>\n"
                )
                running_page += 1
            f.write("  </ul>\n</li>\n")
        f.write("</ul>\n</section>\n")

    def generate_sample_page(self, f, page_idx: int, gname: str, sidx: int, entry: Dict) -> None:
        """Generate a single sample page"""
        f.write(f"<section class='sample-page' id='page{page_idx}'>\n")
        f.write(f"<h2>{escape(gname)} — Sample {sidx+1}</h2>\n")

        # Display initial room image if available
        if self.save_images and entry.get("initial_room_image"):
            img_name = entry["initial_room_image"]
            f.write(f"<img src='{img_name}' class='room' alt='Initial room state'>\n")

        # Environment config
        cfg = entry["env_info"]["config"]
        f.write("<div class='metrics'><strong>🔧 Environment Configuration</strong>")
        f.write(VisualizationHelper.dict_to_html(cfg))
        f.write("</div>\n")

        # Get environment turn logs directly
        env_turn_logs = entry.get("env_turn_logs", [])
        
        # Generate turns from env logs
        for t_idx, env_log in enumerate(env_turn_logs):
            f.write("<div class='turn-split'>\n")
            f.write(f"<h3>🔄 Turn {t_idx+1}</h3>\n")
            
            # Left side: conversation and metrics
            f.write("<div class='turn-left'>\n")
            
            # Display user message (environment observation)
            if env_log['user_message']:
                u_short = escape(env_log['user_message'][:300]).replace("\n", "<br>")
                u_full = escape(env_log['user_message']).replace("\n", "<br>")
                obs_id = f"obs_{page_idx}_{t_idx}"
                f.write(f"<div class='block user expandable' onclick='expandObservation(\"{obs_id}\")'><strong>👤 Environment Observation <span class='expand-hint'>(click to expand)</span></strong><br>{u_short}...</div>\n")
                f.write(f"<div id='{obs_id}' class='observation-full' style='display:none'>{u_full}</div>\n")
                # u_full = escape(env_log['user_message']).replace("\n", "<br>")
                # f.write(f"<div class='block user'><strong>👤 Environment Observation</strong><br>{u_full}</div>\n")
            
            think_content, answer_content = env_log['assistant_think_message'], env_log['assistant_parsed_message']
            # Display think content
            if think_content:
                if False: # len(think_content) > 300:  # Make expandable if long
                    think_short = escape(think_content[:300]).replace("\n", "<br>")
                    think_full = escape(think_content).replace("\n", "<br>")
                    think_id = f"think_{page_idx}_{t_idx}"
                    f.write(f"<div class='block think expandable' onclick='expandThinking(\"{think_id}\")'><strong>🤔 Assistant Thinking <span class='expand-hint'>(click to expand)</span></strong><br>{think_short}...</div>\n")
                    f.write(f"<div id='{think_id}' class='thinking-full' style='display:none'>{think_full}</div>\n")
                else:
                    think = escape(think_content).replace("\n", "<br>")
                    f.write(f"<div class='block think'><strong>🤔 Assistant Thinking</strong><br>{think}</div>\n")
            # Display answer content
            if answer_content:
                answer = escape(answer_content).replace("\n", "<br>")
                f.write(f"<div class='block answer'><strong>💬 Assistant Action</strong><br>{answer}</div>\n")
            
            # Display evaluation information if available
            if not env_log['is_exploration_phase'] and env_log['evaluation_log']:
                eval_log = env_log['evaluation_log']
                f.write("<div class='block evaluation'><strong>✅ Evaluation</strong>")
                
                details = {
                    **eval_log["evaluation_data"],
                    **eval_log.get("evaluation_info", {}),
                    "Correct": eval_log.get("is_correct")
                }
                
                f.write(VisualizationHelper.dict_to_html(details))
                f.write("</div>\n")
            
            # Display cognitive map information if available
            if env_log['cogmap_log']:
                cogmap_log = env_log['cogmap_log']
                f.write("<div class='block cogmap'><strong>🧠 Cognitive Map</strong>")
                
                details = copy.deepcopy(cogmap_log)
                details.pop('pred_room_state')
                f.write(VisualizationHelper.dict_to_html(details))
                f.write("</div>\n")



            # Display turn metrics from env log
            metrics = {}
            if env_log['is_exploration_phase'] and env_log['exploration_log']:
                exp_log = env_log['exploration_log']
                metrics.update({
                    "coverage": exp_log['coverage'],
                    "redundancy": exp_log['redundancy'],
                    "is_redundant": exp_log['is_redundant']
                })
            
            # Add info from env log
            if env_log['info']:
                metrics.update(env_log['info'])
            
            f.write("<div class='metrics'><strong>📈 Turn Metrics</strong>")
            f.write(VisualizationHelper.dict_to_html(metrics))
            f.write("</div>\n")
            f.write("</div>\n")  # End turn-left
            
            # Right side: room and message images
            f.write("<div class='turn-right'>\n")
            if self.show_images:
                # previous image (initial if first turn)
                prev_img = (env_turn_logs[t_idx-1].get('room_image') if t_idx > 0 else entry.get('initial_room_image'))
                if prev_img:
                    f.write(f"<figure><img src='{prev_img}' class='room-plot' alt='Previous state'><figcaption>State before Turn {t_idx+1}</figcaption></figure>\n")
                # current image
                curr_img = env_log.get('room_image')
                if curr_img:
                    f.write(f"<figure><img src='{curr_img}' class='room-plot' alt='Current state'><figcaption>State at Turn {t_idx+1}</figcaption></figure>\n")
            f.write("</div>\n")
            
            # Display message images
            if self.save_images and 'message_images' in env_log:
                for key, images in env_log['message_images'].items():
                    if 'image' in key.lower() and images:
                        for img_idx, img_path in enumerate(images):
                            if isinstance(img_path, str):  # It's a path
                                f.write(f"<img src='{img_path}' class='message-image' alt='Environment image {img_idx+1}'>\n")
            f.write("</div>\n")  # End turn-right
            
            f.write("</div>\n")  # End turn-split

        # Final metrics
        summary = entry.get("summary", {})
        f.write("<div class='metrics'><strong>📊 Sample Final Metrics</strong>")
        f.write(VisualizationHelper.dict_to_html(summary))
        f.write("</div>\n")

        f.write("</section>\n")

    def generate_html(self) -> str:
        """Generate the complete HTML file"""
        with open(self.output_html, "w") as f:
            # Write HTML header with CSS and JS
            f.write(HTML_TEMPLATE.format(
                model_name=escape(self.meta.get('model_name', 'Unknown Model')),
                total_pages=self.total_pages,
                css_styles=CSS_STYLES,
                javascript_code=JAVASCRIPT_CODE
            ))

            # Generate TOC page (now with summaries)
            self.generate_toc_page(f)

            # Generate sample pages
            for page_idx, (gname, sidx, entry) in enumerate(self.flat, start=1):
                self.generate_sample_page(f, page_idx, gname, sidx, entry)

            f.write("</body></html>")

        return self.output_html


class Visualization:
    """Main visualization class for JSON data"""
    
    def __init__(self, json_path: str, output_html: str, save_images: bool = True):
        self.json_path = json_path
        self.output_html = output_html
        self.save_images = save_images

    def load_data(self) -> Dict:
        """Load JSON data from file"""
        with open(self.json_path, "r") as f:
            return json.load(f)

    def visualize(self) -> str:
        """Main method to generate visualization"""
        data = self.load_data()
        generator = HTMLGenerator(data, self.output_html, self.save_images)
        return generator.generate_html()


def visualize_json(json_path: str, output_html: str, save_images: bool = True) -> str:
    viz = Visualization(json_path, output_html, save_images)
    return viz.visualize()


		