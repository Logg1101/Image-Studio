import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
from pathlib import Path

from core.types import GenerationRequest, ModelInfo
from engines.flux.engine import FluxEngine

class TestFluxEngine(unittest.TestCase):
    def setUp(self):
        self.engine = FluxEngine()
        self.dummy_model = ModelInfo(
            id="test_flux",
            architecture="flux",
            variant="schnell",
            format="gguf",
            transformer_path="models/checkpoints/Flux/flux1-schnell-q4.gguf"
        )

    def test_supports(self):
        self.assertTrue(self.engine.supports(self.dummy_model))
        sdxl_model = ModelInfo("sdxl", "sdxl", "base", "safetensors", "")
        self.assertFalse(self.engine.supports(sdxl_model))

    def test_unload_lora_weights_success(self):
        import tempfile
        mock_pipe = MagicMock()
        dummy_img = Image.new("RGB", (64, 64), color="red")
        mock_output = MagicMock()
        mock_output.images = [dummy_img]
        mock_pipe.return_value = mock_output

        self.engine.pipeline = mock_pipe
        self.engine.current_model_info = self.dummy_model

        req = GenerationRequest(
            model=self.dummy_model,
            prompt="test flux",
            width=64,
            height=64,
            steps=1,
            seed=42,
            loras={"dummy_lora.safetensors": 0.8}
        )

        out_tmp = str(Path(tempfile.gettempdir()) / "test_flux_out.png")
        with patch("core.project_manager.ProjectManager.get_output_destination", return_value=out_tmp), \
             patch("core.usage_tracker.UsageTracker.log_generation"):
            res = self.engine.generate(req)

        mock_pipe.load_lora_weights.assert_called_once()
        mock_pipe.set_adapters.assert_called_once()
        mock_pipe.unload_lora_weights.assert_called_once()
        self.assertIsNotNone(self.engine.pipeline)

    def test_unload_lora_weights_failure_clears_pipeline(self):
        import tempfile
        mock_pipe = MagicMock()
        dummy_img = Image.new("RGB", (64, 64), color="blue")
        mock_output = MagicMock()
        mock_output.images = [dummy_img]
        mock_pipe.return_value = mock_output
        mock_pipe.unload_lora_weights.side_effect = RuntimeError("Failed to detach LoRA adapter weights")

        self.engine.pipeline = mock_pipe
        self.engine.current_model_info = self.dummy_model

        req = GenerationRequest(
            model=self.dummy_model,
            prompt="test flux",
            width=64,
            height=64,
            steps=1,
            seed=42,
            loras={"dummy_lora.safetensors": 0.8}
        )

        out_tmp = str(Path(tempfile.gettempdir()) / "test_flux_out_err.png")
        with patch("core.project_manager.ProjectManager.get_output_destination", return_value=out_tmp), \
             patch("core.usage_tracker.UsageTracker.log_generation"):
            res = self.engine.generate(req)

        mock_pipe.unload_lora_weights.assert_called_once()
        mock_pipe.disable_lora.assert_called_once()
        # Pipeline must be cleared to prevent adapter contamination in future runs
        self.assertIsNone(self.engine.pipeline)
        self.assertIsNone(self.engine.current_model_info)

if __name__ == "__main__":
    unittest.main()
