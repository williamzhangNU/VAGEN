"""
Simple image handler for spatial environments
"""
import os
import json
import numpy as np
from typing import Dict, Tuple, Union
from PIL import Image


class ImageHandler:
    """Handle images and data initialization based on position and orientation."""
    
    def __init__(self, base_dir: str, seed: int = None, image_size: tuple = (512, 512), preload_images: bool = True):
        """
        Initialize image handler with data loading.
        
        Args:
            base_dir: Base directory containing data subdirectories
            seed: Random seed for directory selection
            image_size: Target size for loaded images
            preload_images: Whether to load all images into memory
        """
        self.image_size = image_size
        self.preload_images = preload_images
        self.image_dir, self.json_data = self._load_data(base_dir, seed)
        self._name_to_cam = self._build_name_to_cam_mapping(self.json_data)
        self._image_map = self._load_images()
    
    def _load_data(self, base_dir: str, seed: int = None) -> Tuple[str, dict]:
        """Load JSON data from selected subdirectory."""
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        data_idx = (seed % len(subdirs)) if seed else np.random.randint(0, len(subdirs))
        image_dir = os.path.join(base_dir, subdirs[data_idx])
        
        with open(os.path.join(image_dir, "meta_data.json"), 'r') as f:
            json_data = json.load(f)
            
        return image_dir, json_data
    
    def _build_name_to_cam_mapping(self, json_data: dict) -> Dict[str, str]:
        """Map object names to camera IDs by matching positions."""
        name_to_cam = {'central': 'central'}
        
        for obj in json_data.get('objects', []):
            obj_pos = (obj['pos']['x'], obj['pos']['z'])
            obj_name = obj['model']
            
            for cam in json_data.get('cameras', []):
                if cam['id'] == 'central':
                    continue
                cam_pos = (cam['position']['x'], cam['position']['z'])
                if obj_pos == cam_pos:
                    name_to_cam[obj_name] = cam['id']
                    break
        
        return name_to_cam
    
    def _load_images(self) -> Dict[Tuple[str, str], Union[Image.Image, str]]:
        """Load images or paths based on preload setting."""
        image_map = {}
        
        for entry in self.json_data.get('images', []):
            key = (entry['cam_id'], entry['direction'])
            path = os.path.join(self.image_dir, entry['file'])
            
            if os.path.exists(path):
                if self.preload_images:
                    image_map[key] = Image.open(path).resize(self.image_size, Image.LANCZOS)
                else:
                    image_map[key] = path
        
        return image_map
    
    def get_image(self, position: str, direction: str) -> Image.Image:
        """
        Get image for given position and direction.
        
        Args:
            position: Object/position name (e.g., 'central', 'chair_willisau_riale')
            direction: Cardinal direction ('north', 'south', 'east', 'west')
            
        Returns:
            PIL Image
            
        Raises:
            KeyError: If position or image not found
        """
        if position not in self._name_to_cam:
            raise KeyError(f"Position '{position}' not found")
        
        cam_id = self._name_to_cam[position]
        image_key = (cam_id, direction)
        
        if image_key not in self._image_map:
            raise KeyError(f"Image not found for position '{position}' (cam_id: '{cam_id}') facing '{direction}'")
        
        if self.preload_images:
            return self._image_map[image_key]
        else:
            path = self._image_map[image_key]
            return Image.open(path).resize(self.image_size, Image.LANCZOS)