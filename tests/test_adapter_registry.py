import unittest
from pathlib import Path
from adapters.registry import adapter_registry

class TestAdapterRegistry(unittest.TestCase):

    def test_scan_loras(self):
        loras = adapter_registry.scan_loras()
        self.assertIsInstance(loras, dict)
        self.assertGreater(len(loras), 0)
        # Verify extensions are valid
        for name, path in loras.items():
            self.assertIn(path.suffix.lower(), {".safetensors", ".pt", ".bin", ".ckpt"})

    def test_scan_controlnets(self):
        cnets = adapter_registry.scan_controlnets()
        self.assertIsInstance(cnets, dict)

    def test_scan_ip_adapters(self):
        ips = adapter_registry.scan_ip_adapters()
        self.assertIsInstance(ips, dict)

    def test_unload_all_clean_state(self):
        adapter_registry.unload_all(pipeline=None)
        self.assertEqual(len(adapter_registry.active_loras), 0)
        self.assertIsNone(adapter_registry.active_controlnet)
        self.assertIsNone(adapter_registry.active_ip_adapter)
        self.assertIsNone(adapter_registry.active_pulid)

    def test_missing_ip_adapter_raises_generation_error(self):
        import unittest.mock
        import torch
        from core.types import GenerationRequest
        from core.exceptions import GenerationError
        from engines.sdxl.engine import SDXLEngine

        engine = SDXLEngine()
        engine.pipeline = unittest.mock.MagicMock()
        engine.pipeline.device = torch.device("cpu")
        engine.base_scheduler_config = {}
        t = torch.zeros((1, 77, 2048))
        engine.compel_processor = unittest.mock.MagicMock(return_value=(t, t))

        req = GenerationRequest(
            prompt="test",
            model="sd_xl_base_1.0.safetensors",
            ip_adapter_name="nonexistent_ip_adapter_12345",
        )
        with self.assertRaises(GenerationError) as ctx:
            engine.generate(req)
        self.assertIn("nonexistent_ip_adapter_12345", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
