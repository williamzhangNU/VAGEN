import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Optional, List, Any

def calculate_object_distance(pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
    """Calculate 3D distance between two object positions.
    
    Args:
        pos1: First position with 'x', 'y', 'z' keys
        pos2: Second position with 'x', 'y', 'z' keys
        
    Returns:
        Euclidean distance between the positions
    """
    return np.sqrt(
        (pos1['x'] - pos2['x'])**2 +
        (pos1['y'] - pos2['y'])**2 +
        (pos1['z'] - pos2['z'])**2
    )

def calculate_rotation_difference(rot1: Dict[str, float], rot2: Dict[str, float]) -> float:
    """Calculate rotation difference between two orientations.
    
    Args:
        rot1: First rotation with 'x', 'y', 'z' keys
        rot2: Second rotation with 'x', 'y', 'z' keys
        
    Returns:
        Angular difference in degrees
    """
    # Focus on Y rotation (most important for object orientation)
    y_diff = abs(rot1['y'] - rot2['y'])
    # Handle wrap-around (e.g., 359° vs 1°)
    y_diff = min(y_diff, 360 - y_diff)
    return y_diff

def extract_object_memory_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract object memory from agent's response text.
    
    Args:
        text: Agent's response containing object descriptions
        
    Returns:
        List of object memory dictionaries
    """
    memory = []
    
    # Try to extract JSON array
    try:
        import re
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            memory_json = json_match.group()
            memory = json.loads(memory_json)
            return memory
    except:
        pass
    
    # Fallback: parse text descriptions
    lines = text.split('\n')
    for line in lines:
        if 'name:' in line.lower() or 'object:' in line.lower():
            # Simple text parsing for object descriptions
            obj_info = {'name': '', 'type': '', 'position': {}, 'rotation': {}, 'openness': None}
            # This would need more sophisticated parsing based on actual agent output format
            memory.append(obj_info)
    
    return memory

def format_object_memory(objects: List[Dict[str, Any]]) -> str:
    """Format object memory for display or storage.
    
    Args:
        objects: List of object dictionaries
        
    Returns:
        Formatted string representation
    """
    if not objects:
        return "No objects recorded."
    
    formatted = "Recorded Objects:\n"
    for i, obj in enumerate(objects, 1):
        formatted += f"{i}. {obj.get('name', 'Unknown')}\n"
        formatted += f"   Type: {obj.get('type', 'Unknown')}\n"
        
        pos = obj.get('position', {})
        if pos:
            formatted += f"   Position: ({pos.get('x', 0):.2f}, {pos.get('y', 0):.2f}, {pos.get('z', 0):.2f})\n"
        
        rot = obj.get('rotation', {})
        if rot:
            formatted += f"   Rotation: Y={rot.get('y', 0):.1f}°\n"
        
        if obj.get('openness') is not None:
            formatted += f"   Openness: {obj.get('openness'):.2f}\n"
        
        formatted += "\n"
    
    return formatted

def draw_object_boxes(
    image: Image.Image,
    objects: List[Dict[str, Any]],
    instance_detections: Dict[str, np.ndarray],
    output_path: str,
    highlight_objects: Optional[List[str]] = None
):
    """Draw bounding boxes around objects in the image.
    
    Args:
        image: PIL Image to draw on
        objects: List of object dictionaries with names
        instance_detections: Dictionary mapping object IDs to bounding boxes
        output_path: Path to save the annotated image
        highlight_objects: Optional list of object names to highlight
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, obj in enumerate(objects):
        obj_name = obj.get('name', '')
        if obj_name in instance_detections:
            bbox = instance_detections[obj_name]
            color = colors[i % len(colors)]
            
            # Highlight special objects
            if highlight_objects and obj_name in highlight_objects:
                color = (255, 255, 255)  # White for highlighted objects
                width = 3
            else:
                width = 2
            
            # Draw bounding box
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=color, width=width)
            
            # Draw label
            label = obj.get('type', obj_name.split('_')[0])
            draw.text((bbox[0], bbox[1] - 20), label, fill=color, font=font)
    
    image.save(output_path)

def validate_object_poses(poses: List[Dict[str, Any]]) -> bool:
    """Validate object pose data structure.
    
    Args:
        poses: List of object pose dictionaries
        
    Returns:
        True if all poses are valid, False otherwise
    """
    required_keys = ['objectName', 'position', 'rotation']
    position_keys = ['x', 'y', 'z']
    rotation_keys = ['x', 'y', 'z']
    
    for pose in poses:
        # Check required top-level keys
        if not all(key in pose for key in required_keys):
            return False
        
        # Check position structure
        position = pose['position']
        if not all(key in position for key in position_keys):
            return False
        
        # Check rotation structure
        rotation = pose['rotation']
        if not all(key in rotation for key in rotation_keys):
            return False
        
        # Check that values are numeric
        try:
            for key in position_keys:
                float(position[key])
            for key in rotation_keys:
                float(rotation[key])
        except (ValueError, TypeError):
            return False
    
    return True

def create_rearrangement_summary(
    target_objects: Dict[str, Any],
    current_objects: Dict[str, Any],
    success_threshold: float = 0.5
) -> Dict[str, Any]:
    """Create a summary of rearrangement progress.
    
    Args:
        target_objects: Dictionary of target object states
        current_objects: Dictionary of current object states
        success_threshold: Distance threshold for success
        
    Returns:
        Summary dictionary with progress information
    """
    summary = {
        'total_objects': len(target_objects),
        'successful_objects': 0,
        'failed_objects': 0,
        'object_details': [],
        'overall_success_rate': 0.0
    }
    
    for obj_name, target_state in target_objects.items():
        if obj_name in current_objects:
            current_state = current_objects[obj_name]
            
            # Calculate position difference
            pos_diff = calculate_object_distance(
                target_state['position'],
                current_state['position']
            )
            
            # Calculate rotation difference
            rot_diff = calculate_rotation_difference(
                target_state['rotation'],
                current_state['rotation']
            )
            
            success = pos_diff < success_threshold
            if success:
                summary['successful_objects'] += 1
            else:
                summary['failed_objects'] += 1
            
            summary['object_details'].append({
                'name': obj_name,
                'position_difference': pos_diff,
                'rotation_difference': rot_diff,
                'success': success
            })
        else:
            summary['failed_objects'] += 1
            summary['object_details'].append({
                'name': obj_name,
                'position_difference': float('inf'),
                'rotation_difference': float('inf'),
                'success': False,
                'error': 'Object not found'
            })
    
    summary['overall_success_rate'] = (
        summary['successful_objects'] / summary['total_objects'] 
        if summary['total_objects'] > 0 else 0.0
    )
    
    return summary

def generate_rearrangement_report(summary: Dict[str, Any]) -> str:
    """Generate a human-readable rearrangement report.
    
    Args:
        summary: Summary dictionary from create_rearrangement_summary
        
    Returns:
        Formatted report string
    """
    report = f"Rearrangement Task Report\n"
    report += f"========================\n\n"
    report += f"Overall Success Rate: {summary['overall_success_rate']:.2%}\n"
    report += f"Successful Objects: {summary['successful_objects']}/{summary['total_objects']}\n"
    report += f"Failed Objects: {summary['failed_objects']}\n\n"
    
    report += "Object Details:\n"
    report += "---------------\n"
    
    for detail in summary['object_details']:
        status = "✓" if detail['success'] else "✗"
        report += f"{status} {detail['name']}\n"
        
        if 'error' in detail:
            report += f"   Error: {detail['error']}\n"
        else:
            report += f"   Position Diff: {detail['position_difference']:.3f}m\n"
            report += f"   Rotation Diff: {detail['rotation_difference']:.1f}°\n"
        
        report += "\n"
    
    return report
