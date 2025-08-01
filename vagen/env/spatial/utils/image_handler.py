"""
Simple image handler for spatial environments
"""
import os
import json
import numpy as np
from typing import Dict, Union
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
        self.objects = {obj['object_id']: obj for obj in self.json_data.get('objects', [])}
        self._image_map = self._load_images()
        self.name_2_cam_id = {obj['name']: obj['object_id'] for obj in self.objects.values()}
        self.name_2_cam_id['agent'] = 'agent'
    
    def _load_data(self, base_dir: str, seed: int = None) -> tuple:
        """Load JSON data from selected subdirectory."""
        subdirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        data_idx = (seed % len(subdirs)) if seed is not None else np.random.randint(0, len(subdirs))
        image_dir = os.path.join(base_dir, subdirs[data_idx])
        
        with open(os.path.join(image_dir, "meta_data.json"), 'r') as f:
            json_data = json.load(f)
            
        return image_dir, json_data
    
    def _load_images(self) -> Dict[str, Union[Image.Image, str]]:
        """Load images or paths based on preload setting."""
        image_map = {}
        
        for entry in self.json_data.get('images', []):
            key = f"{entry['cam_id']}_facing_{entry['direction']}"
            path = os.path.join(self.image_dir, entry['file'])
            
            if os.path.exists(path):
                if self.preload_images:
                    image_map[key] = Image.open(path).resize(self.image_size, Image.LANCZOS)
                else:
                    image_map[key] = path
        image_map['topdown'] = Image.open(os.path.join(self.image_dir, 'top_down_annotated.png'))
        image_map['oblique'] = Image.open(os.path.join(self.image_dir, 'oblique_view.png'))
        return image_map
    
    def get_image(self, name: str = 'agent', direction: str = 'north') -> Image.Image:
        """
        Get image for given camera ID and direction.
        
        Args:
            name: Name of the object ('agent' or object_name or 'topdown' as string)
            direction: Cardinal direction ('north', 'south', 'east', 'west')
            
        Returns:
            PIL Image
            
        Raises:
            KeyError: If image not found
        """
        key = f"{self.name_2_cam_id[name]}_facing_{direction}" if name != 'topdown' else 'topdown'
        
        if key not in self._image_map:
            raise KeyError(f"Image not found for name '{name}' facing '{direction}'")
        
        if self.preload_images:
            return self._image_map[key]
        else:
            path = self._image_map[key]
            return Image.open(path).resize(self.image_size, Image.LANCZOS)