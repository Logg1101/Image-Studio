import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List
from PIL import Image
import config.paths as paths

class AdapterManager:
    """Manages discovery and preprocessing for ControlNet, IP-Adapter, and PuLID modules."""
    
    def __init__(self):
        paths.ensure_directories()

    SUPPORTED_EXTENSIONS = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx"}

    def scan_controlnets(self) -> Dict[str, str]:
        """Scans models/controlnet/ for subdirectories or model files in any format."""
        result: Dict[str, str] = {}
        if not paths.CONTROLNET_DIR.exists():
            return result
            
        for item in paths.CONTROLNET_DIR.iterdir():
            if item.is_dir():
                result[item.name.lower()] = str(item.resolve())
            elif item.is_file() and item.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                result[item.stem.lower()] = str(item.resolve())
        return result

    def scan_ip_adapters(self) -> Dict[str, str]:
        """Scans models/ipadapter/ for adapter files in any format."""
        result: Dict[str, str] = {}
        if not paths.IPADAPTER_DIR.exists():
            return result
            
        for file_path in paths.IPADAPTER_DIR.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                name = file_path.stem
                result[name] = str(file_path.resolve())
        return result

    def scan_pulid(self) -> Dict[str, str]:
        """Scans models/pulid/ for PuLID models in any format."""
        result: Dict[str, str] = {}
        if not paths.PULID_DIR.exists():
            return result
            
        for file_path in paths.PULID_DIR.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                result[file_path.stem] = str(file_path.resolve())
        return result

    def preprocess_control_image(self, image: Image.Image, control_type: str) -> Image.Image:
        """Applies computer vision preprocessors (Canny, Depth, Lineart, OpenPose) to a control image."""
        if not image or control_type.lower() == "none":
            return image

        img_np = np.array(image.convert("RGB"))
        control_type = control_type.lower()

        if control_type == "canny":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            processed_np = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
        elif control_type == "depth":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(sobelx, sobely)
            normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            processed_np = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
            
        elif control_type == "lineart":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            inverted = cv2.bitwise_not(gray)
            blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
            sketch = cv2.divide(gray, 255 - blurred, scale=256.0)
            processed_np = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
            
        elif control_type == "openpose":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            processed_np = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
            
        else:
            processed_np = img_np

        return Image.fromarray(processed_np)

adapter_manager = AdapterManager()
