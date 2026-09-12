import unittest
from pathlib import Path
import torch

from adapters.base import AdapterType, AdapterStatus
from adapters.lora.detector import LoRAFormatDetector
from adapters.lora.inspector import LoRACheckpointInspector
from adapters.lora.converter import LoRAKeyConverter
from adapters.lora.handler import UniversalLoRAHandler

class TestUniversalLoRASubsystem(unittest.TestCase):

    def setUp(self):
        self.lora_dir = Path("models/loras/characters")

    def test_belfast_format_detection(self):
        belfast_path = self.lora_dir / "Belfast_HassakuXL.safetensors"
        if not belfast_path.exists():
            self.skipTest(f"{belfast_path} not found.")

        info = LoRAFormatDetector.detect(belfast_path)
        self.assertEqual(info.name, "Belfast_HassakuXL")
        self.assertEqual(info.architecture, "sdxl")
        self.assertEqual(info.adapter_type, AdapterType.LYCORIS)
        self.assertEqual(info.rank, 32)
        self.assertEqual(info.alpha, 16.0)
        self.assertEqual(info.tensor_count, 2382)

    def test_akeno_format_detection(self):
        akeno_path = self.lora_dir / "Akeno_HassakuXIllustrious_V1.safetensors"
        if not akeno_path.exists():
            self.skipTest(f"{akeno_path} not found.")

        info = LoRAFormatDetector.detect(akeno_path)
        self.assertEqual(info.name, "Akeno_HassakuXIllustrious_V1")
        self.assertEqual(info.architecture, "sdxl")
        self.assertGreater(info.tensor_count, 2000)

    def test_belfast_inspection_validation(self):
        belfast_path = self.lora_dir / "Belfast_HassakuXL.safetensors"
        if not belfast_path.exists():
            self.skipTest(f"{belfast_path} not found.")

        res = LoRACheckpointInspector.inspect(belfast_path, target_architecture="sdxl")
        self.assertTrue(res.is_valid)
        self.assertEqual(res.status, AdapterStatus.LOADED)
        self.assertEqual(res.diagnostics["total_tensors"], 2382)
        self.assertEqual(res.diagnostics["add_embedding_tensors"], 6)

    def test_key_normalization(self):
        fake_state_dict = {
            "lora_unet_add_embedding_linear_1.alpha": torch.tensor(16.0),
            "lora_unet_add_embedding_linear_1.lora_down.weight": torch.randn(32, 2816),
            "lora_unet_conv_in.lora_down.weight": torch.randn(32, 4),
            "lycoris_unet_mid_block_attentions_0_to_q.lora_down.weight": torch.randn(32, 640),
            "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": torch.randn(32, 768),
        }

        normalized, stats = LoRAKeyConverter.normalize_state_dict(fake_state_dict, target_architecture="sdxl")
        self.assertIn("lora_unet_conv_in.lora_down.weight", normalized)
        self.assertIn("lora_unet_mid_block_attentions_0_to_q.lora_down.weight", normalized)
        # add_embedding and text_encoder weights are filtered cleanly
        self.assertNotIn("lora_unet_add_embedding_linear_1.lora_down.weight", normalized)
        self.assertNotIn("lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight", normalized)
        self.assertEqual(stats["normalized_tensors"], 2)
        self.assertEqual(stats["filtered_optional_tensors"], 3)

    def test_set_strength_preserves_multiple_active_adapters(self):
        """Verify adjusting one adapter's strength preserves all other active adapters."""
        from unittest.mock import MagicMock, patch
        from adapters.registry import adapter_registry
        from adapters.base import AdapterInfo, AdapterType

        mock_pipe = MagicMock()
        mock_pipe.set_adapters = MagicMock()
        mock_info = AdapterInfo(
            name="mock", file_path=Path("mock.safetensors"), adapter_type=AdapterType.LORA, architecture="sdxl"
        )

        with patch("adapters.lora.handler.LoRAFormatDetector.detect", return_value=mock_info):
            handler_a = UniversalLoRAHandler(name="char_a", file_path="dummy_a.safetensors")
            handler_a.is_attached = True
            handler_a.strength = 0.8

            handler_b = UniversalLoRAHandler(name="style_b", file_path="dummy_b.safetensors")
            handler_b.is_attached = True
            handler_b.strength = 0.6

        # Register both in adapter registry
        adapter_registry.active_loras = {"char_a": handler_a, "style_b": handler_b}

        try:
            # Adjust strength on handler_a
            handler_a.set_strength(0.95, mock_pipe)

            # Assert pipeline.set_adapters called with both adapters intact
            mock_pipe.set_adapters.assert_called_once()
            call_args = mock_pipe.set_adapters.call_args
            names_passed, kwargs_passed = call_args[0], call_args[1]
            self.assertEqual(names_passed[0], ["char_a", "style_b"])
            self.assertEqual(kwargs_passed["adapter_weights"], [0.95, 0.6])
        finally:
            adapter_registry.active_loras.clear()

    def test_adapter_registry_set_adapter_strength(self):
        """Verify adapter_registry.set_adapter_strength updates weight and maintains all adapters."""
        from unittest.mock import MagicMock, patch
        from adapters.registry import adapter_registry
        from adapters.base import AdapterInfo, AdapterType

        mock_pipe = MagicMock()
        mock_info = AdapterInfo(
            name="mock", file_path=Path("mock.safetensors"), adapter_type=AdapterType.LORA, architecture="sdxl"
        )

        with patch("adapters.lora.handler.LoRAFormatDetector.detect", return_value=mock_info):
            handler_a = UniversalLoRAHandler(name="char_a", file_path="dummy_a.safetensors")
            handler_a.strength = 0.8
            handler_b = UniversalLoRAHandler(name="style_b", file_path="dummy_b.safetensors")
            handler_b.strength = 0.6

        adapter_registry.active_loras = {"char_a": handler_a, "style_b": handler_b}
        try:
            res = adapter_registry.set_adapter_strength("style_b", 0.4, mock_pipe)
            self.assertTrue(res)
            self.assertEqual(handler_b.strength, 0.4)
            mock_pipe.set_adapters.assert_called_once_with(["char_a", "style_b"], adapter_weights=[0.8, 0.4])

            # Non-existent adapter returns False
            self.assertFalse(adapter_registry.set_adapter_strength("unknown_xyz", 0.5, mock_pipe))
        finally:
            adapter_registry.active_loras.clear()


if __name__ == "__main__":
    unittest.main()
