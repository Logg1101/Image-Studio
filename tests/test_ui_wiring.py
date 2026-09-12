import unittest
from ui.app import create_ui, get_random_prompt_text, enhance_prompt_text


class TestUIWiring(unittest.TestCase):
    def test_ui_initialization_and_cancellation_wiring(self):
        """Verify UI builds cleanly and stop buttons register job cancellation hooks."""
        app = create_ui()
        self.assertIsNotNone(app)

        cancellers = [k for k, v in app.fns.items() if getattr(v, "cancels", None)]
        self.assertEqual(len(cancellers), 2, "Expected 2 cancellation handlers (T2I and I2I)")

        for c in cancellers:
            self.assertGreater(len(app.fns[c].cancels), 0, "Cancel handler must have target job IDs")

    def test_random_prompt_generation(self):
        """Verify random prompt generation produces non-empty prompts for SDXL and Flux."""
        sdxl_prompt = get_random_prompt_text("Illustrious-XL-v1.0")
        self.assertIsInstance(sdxl_prompt, str)
        self.assertGreater(len(sdxl_prompt), 10)

        flux_prompt = get_random_prompt_text("flux-schnell")
        self.assertIsInstance(flux_prompt, str)
        self.assertGreater(len(flux_prompt), 10)

    def test_enhance_prompt_text(self):
        """Verify prompt enhancement functions as expected."""
        base_prompt = "1girl, smiling"
        enhanced_sdxl = enhance_prompt_text(base_prompt, "Illustrious-XL-v1.0")
        self.assertIn("1girl, smiling", enhanced_sdxl)
        self.assertGreater(len(enhanced_sdxl), len(base_prompt))


if __name__ == "__main__":
    unittest.main()
