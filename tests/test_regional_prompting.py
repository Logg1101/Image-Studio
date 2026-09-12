import unittest
import torch
from core.types import CharacterRegion, GenerationRequest, ModelInfo
from pipelines.regional_prompting import get_default_regions, create_regional_masks

class TestRegionalPrompting(unittest.TestCase):
    def test_default_regions(self):
        r2 = get_default_regions(2)
        self.assertEqual(len(r2), 2)
        self.assertEqual(r2[0], (0.0, 0.0, 0.5, 1.0))
        self.assertEqual(r2[1], (0.5, 0.0, 1.0, 1.0))

        r3 = get_default_regions(3)
        self.assertEqual(len(r3), 3)

        r4 = get_default_regions(4)
        self.assertEqual(len(r4), 4)

    def test_regional_mask_shapes_and_normalization(self):
        chars = [
            CharacterRegion(prompt="1girl, blonde", box=(0.0, 0.0, 0.5, 1.0)),
            CharacterRegion(prompt="1boy, black hair", box=(0.5, 0.0, 1.0, 1.0)),
        ]
        masks, base_mask = create_regional_masks(
            latent_h=64,
            latent_w=64,
            characters=chars,
            device="cpu",
            dtype=torch.float32
        )
        self.assertEqual(len(masks), 2)
        self.assertEqual(masks[0].shape, (1, 1, 64, 64))
        self.assertEqual(masks[1].shape, (1, 1, 64, 64))
        self.assertEqual(base_mask.shape, (1, 1, 64, 64))

        # Check that masks are within [0, 1]
        self.assertTrue(torch.all(masks[0] >= 0.0) and torch.all(masks[0] <= 1.0))
        self.assertTrue(torch.all(masks[1] >= 0.0) and torch.all(masks[1] <= 1.0))
        self.assertTrue(torch.all(base_mask >= 0.0) and torch.all(base_mask <= 1.0))

        # Check total composite sum across space is close to 1.0
        total = masks[0] + masks[1] + base_mask
        self.assertTrue(torch.all(total <= 1.001))
        # Center of left half should be ~1.0 for char 0
        self.assertAlmostEqual(masks[0][0, 0, 32, 16].item(), 1.0, places=2)
        # Center of right half should be ~1.0 for char 1
        self.assertAlmostEqual(masks[1][0, 0, 32, 48].item(), 1.0, places=2)
        # Char 0 should have ~0 in right half
        self.assertAlmostEqual(masks[0][0, 0, 32, 48].item(), 0.0, places=2)

    def test_generation_request_characters_compatibility(self):
        m_info = ModelInfo(
            id="dummy", architecture="sdxl", variant="fp16", format="safetensors", transformer_path="dummy.safetensors"
        )
        # Standard request without characters
        req1 = GenerationRequest(model=m_info, prompt="a castle")
        self.assertEqual(req1.characters, [])

        # Multi-character request
        req2 = GenerationRequest(
            model=m_info,
            prompt="shared fantasy landscape",
            characters=[
                CharacterRegion(prompt="warrior", box=(0.0, 0.0, 0.5, 1.0)),
                CharacterRegion(prompt="mage", box=(0.5, 0.0, 1.0, 1.0))
            ]
        )
        self.assertEqual(len(req2.characters), 2)
        self.assertEqual(req2.characters[0].prompt, "warrior")

    def test_activate_single_lora(self):
        from pipelines.regional_prompting import _activate_single_lora
        from unittest.mock import MagicMock

        pipeline = MagicMock()
        _activate_single_lora(pipeline, "char_a", 0.75)

        pipeline.enable_lora.assert_called_once()
        pipeline.set_adapters.assert_called_once_with(["char_a"], adapter_weights=[0.75])

    def test_disable_active_loras(self):
        from pipelines.regional_prompting import _disable_active_loras
        from unittest.mock import MagicMock

        pipeline = MagicMock()
        _disable_active_loras(pipeline)

        pipeline.disable_lora.assert_called_once()
        pipeline.set_adapters.assert_called_once_with([], adapter_weights=[])

    def test_disable_active_loras_fallback_to_zero_weights(self):
        from pipelines.regional_prompting import _disable_active_loras
        from unittest.mock import MagicMock

        pipeline = MagicMock()
        # Make empty list call fail, triggering fallback
        def mock_set_adapters(names, adapter_weights):
            if not names:
                raise ValueError("Empty adapter names not allowed")

        pipeline.set_adapters.side_effect = mock_set_adapters
        pipeline.get_active_adapters.return_value = ["char_a", "char_b"]

        _disable_active_loras(pipeline)
        pipeline.disable_lora.assert_called_once()
        pipeline.get_active_adapters.assert_called_once()

    def test_adapter_helpers_handle_exceptions_gracefully(self):
        from pipelines.regional_prompting import _activate_single_lora, _disable_active_loras
        from unittest.mock import MagicMock

        pipeline = MagicMock()
        pipeline.enable_lora.side_effect = RuntimeError("GPU memory error")
        pipeline.set_adapters.side_effect = ValueError("Invalid adapter config")
        pipeline.disable_lora.side_effect = AttributeError("Missing module")

        # Must not raise exceptions
        _activate_single_lora(pipeline, "broken_lora", 0.5)
        _disable_active_loras(pipeline)


if __name__ == "__main__":
    unittest.main()
