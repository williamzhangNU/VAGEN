#!/usr/bin/env python3
"""
Visualize debug information from mental-rotation inference as HTML.
Usage: python visualize_debug.py --debug_dir debug_results --output_dir debug_html
"""

import os
import json
import argparse
import base64
from pathlib import Path
from typing import Dict, List, Any

def load_debug_data(debug_dir: str) -> List[Dict[str, Any]]:
    """Load all debug JSON files from the debug directory."""
    debug_data = []
    debug_path = Path(debug_dir)
    
    if not debug_path.exists():
        print(f"Debug directory {debug_dir} does not exist!")
        return []
    
    # Find all JSON files
    json_files = list(debug_path.glob("*.json"))
    print(f"Found {len(json_files)} debug files")
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                debug_data.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return debug_data

def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string for HTML embedding."""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return ""

def generate_html_for_env(env_data: Dict[str, Any], debug_dir: str) -> str:
    """Generate HTML content for a single environment."""
    env_id = env_data.get("env_id", "unknown")
    config_id = env_data.get("config_id", "unknown")
    recordings = env_data.get("recordings", [])
    env_states = env_data.get("env_states", {})
    final_metrics = env_data.get("final_metrics", {})
    
    # Check for images directory
    image_dir = Path(debug_dir) / f"{env_id}_images"
    images = {}
    if image_dir.exists():
        for img_file in image_dir.glob("*.png"):
            step_name = img_file.stem  # e.g., "current_step_0", "target_step_0"
            images[step_name] = encode_image_to_base64(str(img_file))
    
    html = f"""
    <div class="environment" id="{env_id}">
        <h2>Environment: {env_id}</h2>
        <div class="env-info">
            <p><strong>Config ID:</strong> {config_id}</p>
            <p><strong>Final Steps:</strong> {final_metrics.get('step', 0)}</p>
            <p><strong>Success:</strong> {final_metrics.get('success', False)}</p>
            <p><strong>Score:</strong> {final_metrics.get('score', 0):.2f}</p>
        </div>
        
        <div class="conversation">
            <h3>Conversation History</h3>
    """
    
    # Process conversation
    step_images = {}  # Track images for each step
    for i, msg in enumerate(recordings):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        # Extract step information from content
        step_num = None
        if "Step" in content and "Result" in content:
            try:
                step_num = int(content.split("Step")[1].split("Result")[0].strip())
            except:
                pass
        
        # Role-specific styling
        role_class = f"message {role}"
        
        html += f"""
            <div class="{role_class}">
                <div class="role">{role.title()}</div>
                <div class="content">
                    <pre>{content}</pre>
        """
        
        # Add images if this message has multi_modal_data
        if "multi_modal_data" in msg:
            html += '<div class="images">'
            mm_data = msg["multi_modal_data"]
            
            for key, values in mm_data.items():
                if "<image>" in key.lower() or "image" in key.lower():
                    html += f'<div class="image-group"><h4>{key}</h4>'
                    
                    # Determine image type and find corresponding saved images
                    if key == "<image>":
                        # Current object image
                        img_key = f"current_step_{step_num if step_num is not None else 0}"
                        if img_key in images:
                            html += f'''
                                <img src="data:image/png;base64,{images[img_key]}" 
                                     alt="Current Object - Step {step_num}" 
                                     class="debug-image current-image"
                                     onclick="openImageModal(this)">
                            '''
                    elif key == "<target_image>":
                        # Target object image
                        img_key = f"target_step_{step_num if step_num is not None else 0}"
                        if img_key in images:
                            html += f'''
                                <img src="data:image/png;base64,{images[img_key]}" 
                                     alt="Target Object - Step {step_num}" 
                                     class="debug-image target-image"
                                     onclick="openImageModal(this)">
                            '''
                    
                    html += '</div>'
            html += '</div>'
        
        html += """
                </div>
            </div>
        """
    
    # Add environment state information
    html += f"""
        </div>
        
        <div class="env-state">
            <h3>Environment State</h3>
            <div class="state-info">
                <p><strong>Done:</strong> {env_states.get('done', False)}</p>
                <p><strong>Step Count:</strong> {env_states.get('step', 0)}</p>
                <p><strong>Total Rewards:</strong> {env_states.get('rewards', [])}</p>
            </div>
        </div>
        
        <div class="metrics">
            <h3>Final Metrics</h3>
            <div class="metrics-grid">
    """
    
    # Display metrics
    for key, value in final_metrics.items():
        if isinstance(value, (int, float)):
            html += f'<div class="metric"><span class="key">{key}:</span> <span class="value">{value:.3f}</span></div>'
        else:
            html += f'<div class="metric"><span class="key">{key}:</span> <span class="value">{value}</span></div>'
    
    html += """
            </div>
        </div>
    </div>
    """
    
    return html

def generate_full_html(debug_data: List[Dict[str, Any]], debug_dir: str) -> str:
    """Generate complete HTML document."""
    
    # Generate navigation
    nav_html = "<ul>"
    for env_data in debug_data:
        env_id = env_data.get("env_id", "unknown")
        success = env_data.get("final_metrics", {}).get("success", False)
        status_class = "success" if success else "failure"
        nav_html += f'<li><a href="#{env_id}" class="{status_class}">{env_id}</a></li>'
    nav_html += "</ul>"
    
    # Generate content for each environment
    content_html = ""
    for env_data in debug_data:
        content_html += generate_html_for_env(env_data, debug_dir)
    
    # Complete HTML document
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Mental Rotation Debug Visualization</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
            }}
            
            .header {{
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                text-align: center;
            }}
            
            .navigation {{
                background-color: #34495e;
                padding: 10px;
                position: sticky;
                top: 0;
                z-index: 100;
            }}
            
            .navigation ul {{
                list-style: none;
                margin: 0;
                padding: 0;
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }}
            
            .navigation li a {{
                color: white;
                text-decoration: none;
                padding: 8px 16px;
                border-radius: 4px;
                background-color: #4a5f7a;
            }}
            
            .navigation li a:hover {{
                background-color: #5a6f8a;
            }}
            
            .navigation li a.success {{
                background-color: #27ae60;
            }}
            
            .navigation li a.failure {{
                background-color: #e74c3c;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            .environment {{
                background-color: white;
                margin-bottom: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            
            .environment h2 {{
                background-color: #3498db;
                color: white;
                margin: 0;
                padding: 15px 20px;
            }}
            
            .env-info {{
                padding: 15px 20px;
                background-color: #ecf0f1;
                border-bottom: 1px solid #ddd;
            }}
            
            .env-info p {{
                margin: 5px 0;
            }}
            
            .conversation {{
                padding: 20px;
            }}
            
            .message {{
                margin-bottom: 15px;
                border-left: 4px solid #ddd;
                padding-left: 15px;
            }}
            
            .message.system {{
                border-left-color: #9b59b6;
            }}
            
            .message.user {{
                border-left-color: #3498db;
            }}
            
            .message.assistant {{
                border-left-color: #e67e22;
            }}
            
            .role {{
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 5px;
                text-transform: uppercase;
                font-size: 12px;
            }}
            
            .content pre {{
                white-space: pre-wrap;
                word-wrap: break-word;
                margin: 0;
                font-family: inherit;
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 4px;
            }}
            
            .images {{
                margin-top: 10px;
            }}
            
            .image-group {{
                margin-bottom: 10px;
            }}
            
            .image-group h4 {{
                margin: 5px 0;
                color: #7f8c8d;
                font-size: 14px;
            }}
            
            .debug-image {{
                max-width: 300px;
                max-height: 200px;
                border: 3px solid #ddd;
                border-radius: 4px;
                margin: 5px;
                cursor: pointer;
                transition: transform 0.2s;
            }}
            
            .debug-image:hover {{
                transform: scale(1.05);
            }}
            
            .debug-image.current-image {{
                border-color: #3498db;
            }}
            
            .debug-image.current-image:hover {{
                border-color: #2980b9;
            }}
            
            .debug-image.target-image {{
                border-color: #e74c3c;
            }}
            
            .debug-image.target-image:hover {{
                border-color: #c0392b;
            }}
            
            .env-state, .metrics {{
                padding: 20px;
                border-top: 1px solid #eee;
            }}
            
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 10px;
            }}
            
            .metric {{
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 4px;
            }}
            
            .metric .key {{
                font-weight: bold;
                color: #2c3e50;
            }}
            
            .metric .value {{
                color: #27ae60;
            }}
            
            /* Modal for image viewing */
            .modal {{
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.8);
            }}
            
            .modal-content {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                max-width: 90%;
                max-height: 90%;
            }}
            
            .modal img {{
                width: 100%;
                height: 100%;
                object-fit: contain;
            }}
            
            .close {{
                position: absolute;
                top: 15px;
                right: 35px;
                color: #f1f1f1;
                font-size: 40px;
                font-weight: bold;
                cursor: pointer;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Mental Rotation Debug Visualization</h1>
            <p>Generated from debug data with {len(debug_data)} environments</p>
        </div>
        
        <div class="navigation">
            {nav_html}
        </div>
        
        <div class="container">
            {content_html}
        </div>
        
        <!-- Modal for image viewing -->
        <div id="imageModal" class="modal">
            <span class="close" onclick="closeImageModal()">&times;</span>
            <div class="modal-content">
                <img id="modalImage" src="" alt="Modal Image">
            </div>
        </div>
        
        <script>
            function openImageModal(img) {{
                document.getElementById('imageModal').style.display = 'block';
                document.getElementById('modalImage').src = img.src;
            }}
            
            function closeImageModal() {{
                document.getElementById('imageModal').style.display = 'none';
            }}
            
            // Close modal when clicking outside of image
            window.onclick = function(event) {{
                var modal = document.getElementById('imageModal');
                if (event.target == modal) {{
                    modal.style.display = 'none';
                }}
            }}
        </script>
    </body>
    </html>
    """
    
    return html

def main():
    parser = argparse.ArgumentParser(description="Visualize mental-rotation debug information as HTML")
    parser.add_argument("--debug_dir", default="debug_results", help="Directory containing debug JSON files")
    parser.add_argument("--output_dir", default="debug_html", help="Output directory for HTML files")
    parser.add_argument("--output_file", default="debug_visualization.html", help="Output HTML filename")
    
    args = parser.parse_args()
    
    # Load debug data
    print(f"Loading debug data from {args.debug_dir}...")
    debug_data = load_debug_data(args.debug_dir)
    
    if not debug_data:
        print("No debug data found!")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate HTML
    print(f"Generating HTML for {len(debug_data)} environments...")
    html_content = generate_full_html(debug_data, args.debug_dir)
    
    # Save HTML file
    output_file = output_dir / args.output_file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ HTML visualization saved to: {output_file}")
    print(f"✓ Open the file in your browser to view the debug information")

if __name__ == "__main__":
    main()
