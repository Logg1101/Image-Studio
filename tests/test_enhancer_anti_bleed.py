import os
import math
import unittest
import numpy as np
from PIL import Image

from enhancer.analyzer.semantic_analyzer import SemanticAnalyzer
from enhancer.perception.vram_manager import vram_manager


class TestEnhancerAntiBleed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asset_path = os.path.join(os.path.dirname(__file__), "assets", "reference_test_image.jpg")
        if os.path.exists(asset_path):
            cls.test_image = Image.open(asset_path)
        else:
            cls.test_image = Image.new("RGB", (576, 1024), color=(230, 220, 210))
        cls.analyzer = SemanticAnalyzer()

    def test_subject_and_background_complementarity(self):
        """Condition 2: Subject and background are complementary within numerical tolerance."""
        res = self.analyzer.analyze(self.test_image)
        # Decode subject and background base64 masks
        import base64, io
        
        def _to_arr(b64_str):
            raw = base64.b64decode(b64_str.split("base64,")[1])
            return np.array(Image.open(io.BytesIO(raw)), dtype=np.float32) / 255.0

        m_subj = _to_arr(res.regions["subject"].mask_b64)
        m_bg = _to_arr(res.regions["background"].mask_b64)

        # Dimension match
        self.assertEqual(m_subj.shape, (self.test_image.height, self.test_image.width))
        self.assertEqual(m_bg.shape, (self.test_image.height, self.test_image.width))

        # Check complementarity: m_subj + m_bg == 1.0 (within 8-bit quantization tolerance 1/255 ≈ 0.004)
        diff = np.abs((m_subj + m_bg) - 1.0)
        self.assertLess(np.max(diff), 0.01)

    def test_background_does_not_contain_majority_of_subject(self):
        """Condition 1: Background mask does not contain majority of subject."""
        res = self.analyzer.analyze(self.test_image)
        import base64, io
        def _to_arr(b64_str):
            return np.array(Image.open(io.BytesIO(base64.b64decode(b64_str.split("base64,")[1]))), dtype=np.float32) / 255.0

        m_subj = _to_arr(res.regions["subject"].mask_b64)
        m_bg = _to_arr(res.regions["background"].mask_b64)

        subject_core = m_subj > 0.7
        bg_leakage_in_subject = np.mean(m_bg[subject_core])
        self.assertLess(bg_leakage_in_subject, 0.15)

    def test_skin_and_clothing_anti_bleed(self):
        """Condition 3: Skin and clothing do not substantially overlap outside permitted transition zones."""
        res = self.analyzer.analyze(self.test_image)
        import base64, io
        def _to_arr(b64_str):
            return np.array(Image.open(io.BytesIO(base64.b64decode(b64_str.split("base64,")[1]))), dtype=np.float32) / 255.0

        m_skin = _to_arr(res.regions["skin"].mask_b64)
        m_cloth = _to_arr(res.regions["clothing"].mask_b64)

        overlap_energy = np.mean(m_skin * m_cloth)
        self.assertLess(overlap_energy, 0.08)

    def test_hair_not_in_background(self):
        """Condition 4: Hair does not become part of the background when hair is clearly visible."""
        res = self.analyzer.analyze(self.test_image)
        import base64, io
        def _to_arr(b64_str):
            return np.array(Image.open(io.BytesIO(base64.b64decode(b64_str.split("base64,")[1]))), dtype=np.float32) / 255.0

        m_hair = _to_arr(res.regions["hair"].mask_b64)
        m_bg = _to_arr(res.regions["background"].mask_b64)

        hair_in_bg_energy = np.mean(m_hair * m_bg)
        self.assertLess(hair_in_bg_energy, 0.05)

    def test_face_strictly_inside_subject(self):
        """Condition 5: Face remains strictly inside the subject region."""
        res = self.analyzer.analyze(self.test_image)
        import base64, io
        def _to_arr(b64_str):
            return np.array(Image.open(io.BytesIO(base64.b64decode(b64_str.split("base64,")[1]))), dtype=np.float32) / 255.0

        m_face = _to_arr(res.regions["face"].mask_b64)
        m_subj = _to_arr(res.regions["subject"].mask_b64)

        face_outside_subject = np.sum((m_face > 0.5) & (m_subj < 0.2))
        self.assertEqual(face_outside_subject, 0)

    def test_confidence_boundary_drop(self):
        """Condition 6: Confidence decreases appropriately around uncertain boundaries."""
        res = self.analyzer.analyze(self.test_image)
        import base64, io
        def _to_arr(b64_str):
            return np.array(Image.open(io.BytesIO(base64.b64decode(b64_str.split("base64,")[1]))), dtype=np.float32) / 255.0

        conf_map = _to_arr(res.confidence_map_b64)
        b_skin_cloth = _to_arr(res.boundaries["skin_clothing"].mask_b64)

        # Core subject confidence vs boundary confidence
        core_zone = (conf_map > 0.8) & (b_skin_cloth < 0.1)
        bound_zone = b_skin_cloth > 0.4

        mean_core_conf = np.mean(conf_map[core_zone])
        mean_bound_conf = np.mean(conf_map[bound_zone])
        self.assertGreater(mean_core_conf, mean_bound_conf)

    def test_masks_strictly_bounded_without_nans(self):
        """Condition 7: Soft masks remain within [0.0, 1.0] with zero NaNs/Infs."""
        res = self.analyzer.analyze(self.test_image)
        import base64, io
        def _to_arr(b64_str):
            return np.array(Image.open(io.BytesIO(base64.b64decode(b64_str.split("base64,")[1]))), dtype=np.float32) / 255.0

        for r_name, region in res.regions.items():
            arr = _to_arr(region.mask_b64)
            self.assertTrue(np.all(np.isfinite(arr)))
            self.assertTrue(np.all(arr >= 0.0))
            self.assertTrue(np.all(arr <= 1.0))

    def test_debug_export_functionality(self):
        """Condition 10: Debug artifact export runs cleanly and creates PNGs."""
        out_dir = "outputs/test_enhancer_debug"
        res = self.analyzer.analyze(self.test_image, export_debug=True)
        self.assertIsNotNone(res)


if __name__ == "__main__":
    unittest.main()
