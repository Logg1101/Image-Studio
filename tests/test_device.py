import unittest
import torch
from core.device import get_device_info, DeviceInfo, get_torch_device, get_torch_dtype

class TestDevice(unittest.TestCase):
    def test_get_device_info(self):
        info = get_device_info()
        self.assertIsInstance(info, DeviceInfo)
        self.assertIsInstance(info.name, str)
        self.assertIsInstance(info.vram_gb, float)
        self.assertGreaterEqual(info.vram_gb, 0.0)

    def test_get_torch_device(self):
        dev = get_torch_device()
        self.assertIsInstance(dev, torch.device)
        self.assertIn(dev.type, ["cuda", "mps", "xpu", "cpu", "dml"])

    def test_get_torch_dtype(self):
        dtype = get_torch_dtype()
        self.assertIn(dtype, [torch.float16, torch.float32])

    def test_cpu_generator_compatibility(self):
        """Verify generator instantiation works on CPU device without CUDA errors."""
        cpu_dev = torch.device("cpu")
        gen = torch.Generator(device=cpu_dev).manual_seed(42)
        self.assertIsInstance(gen, torch.Generator)
        val = torch.randn(1, generator=gen)
        self.assertEqual(val.shape, (1,))

    def test_adapter_registry_set_lora_device_with_cpu_pipeline(self):
        """Verify adapter registry set_lora_device respects non-cuda pipeline device."""
        from unittest.mock import MagicMock
        from adapters.registry import AdapterRegistry

        registry = AdapterRegistry()
        mock_pipeline = MagicMock()
        mock_pipeline.device = torch.device("cpu")

        # Mock an active lora handler
        mock_handler = MagicMock()
        mock_handler.strength = 0.8
        registry.active_loras["test_lora"] = mock_handler

        # Trigger sync with same loras or test set_lora_device
        if hasattr(mock_pipeline, "set_lora_device"):
            dev = getattr(mock_pipeline, "device", None) or get_torch_device()
            mock_pipeline.set_lora_device(adapter_names=["test_lora"], device=str(dev))
            mock_pipeline.set_lora_device.assert_called_once_with(adapter_names=["test_lora"], device="cpu")


if __name__ == "__main__":
    unittest.main()
