import unittest
import tempfile
from pathlib import Path
import torch
import safetensors.torch

from core.model_manager import ModelManager
from core.models.detector import ModelArchitectureDetector

class TestModels(unittest.TestCase):
    def test_scan_models(self):
        manager = ModelManager()
        self.assertIsInstance(manager.available_models, dict)

    def test_detector_flux_schnell_gguf(self):
        p = Path("models/checkpoints/Flux/flux1-schnell-q4.gguf")
        if not p.exists():
            self.skipTest("flux1-schnell-q4.gguf not found")
        info = ModelArchitectureDetector.detect(p)
        self.assertEqual(info["architecture"], "flux")
        self.assertEqual(info["variant"], "schnell")
        self.assertEqual(info["format"], "gguf")

    def test_detector_flux_dev_gguf(self):
        p = Path("models/checkpoints/Flux/modern_anime_full_Q4_0.gguf")
        if not p.exists():
            self.skipTest("modern_anime_full_Q4_0.gguf not found")
        info = ModelArchitectureDetector.detect(p)
        self.assertEqual(info["architecture"], "flux")
        self.assertEqual(info["variant"], "dev")
        self.assertEqual(info["format"], "gguf")

    def test_detector_sdxl_safetensors(self):
        p = Path("models/checkpoints/SDXL/DiamondForge_XL_V4.safetensors")
        if not p.exists():
            self.skipTest("DiamondForge_XL_V4.safetensors not found")
        info = ModelArchitectureDetector.detect(p)
        self.assertEqual(info["architecture"], "sdxl")
        self.assertEqual(info["variant"], "base")
        self.assertEqual(info["format"], "safetensors")

    def test_detector_safetensors_keys_beyond_50(self):
        tmp = Path(tempfile.gettempdir()) / "test_late_flux_key.safetensors"
        try:
            # Create 60 dummy prefix keys
            d = {f"prefix.text_encoder.{i}.weight": torch.zeros((2, 2)) for i in range(60)}
            # Insert Flux marker after index 50
            d["double_blocks.0.img_attn.proj.weight"] = torch.zeros((2, 2))
            safetensors.torch.save_file(d, str(tmp))

            info = ModelArchitectureDetector.detect(tmp)
            self.assertEqual(info["architecture"], "flux")
            self.assertEqual(info["variant"], "schnell")
        finally:
            if tmp.exists():
                tmp.unlink()

    def test_detector_nonexistent_path(self):
        info = ModelArchitectureDetector.detect(Path("nonexistent_model.safetensors"))
        self.assertEqual(info["architecture"], "unknown")
        self.assertEqual(info["format"], "unknown")

if __name__ == "__main__":
    unittest.main()
