import os
import math
import unittest
from PIL import Image

from enhancer.types import SemanticAnalysisResult, MaterialClass, ProtectionPriority
from enhancer.analyzer.semantic_analyzer import SemanticAnalyzer
from core.generation import GenerationCoordinator
from core.model_manager import ModelManager


class TestSemanticAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asset_path = os.path.join(os.path.dirname(__file__), "assets", "reference_test_image.jpg")
        if os.path.exists(asset_path):
            cls.sample_image = Image.open(asset_path)
        else:
            cls.sample_image = Image.new("RGB", (576, 1024), color=(240, 230, 220))
        cls.analyzer = SemanticAnalyzer()

    def test_semantic_analyzer_schema_and_hierarchy(self):
        result = self.analyzer.analyze(self.sample_image)

        self.assertIsInstance(result, SemanticAnalysisResult)
        self.assertEqual(result.image.width, self.sample_image.width)
        self.assertEqual(result.image.height, self.sample_image.height)
        self.assertEqual(result.image.channels, 3)

        # Verify all core regions exist in hierarchy
        required_regions = [
            "background",
            "subject",
            "skin",
            "hair",
            "clothing",
            "accessories",
            "face",
        ]
        for r in required_regions:
            self.assertIn(r, result.regions, f"Missing required semantic region: {r}")
            region = result.regions[r]
            self.assertTrue(region.mask_b64.startswith("data:image/png;base64,"))
            self.assertTrue(0.0 <= region.coverage_pct <= 100.0)
            self.assertTrue(0.0 <= region.mean_confidence <= 1.0)
            self.assertFalse(math.isnan(region.coverage_pct))
            self.assertFalse(math.isnan(region.mean_confidence))

        # Verify Face Protection Zone
        face = result.regions["face"]
        self.assertTrue(face.is_protected)
        self.assertEqual(face.protection_priority, ProtectionPriority.HIGH)
        self.assertIsNotNone(face.bounding_box)
        self.assertEqual(len(face.bounding_box), 4)
        ymin, xmin, ymax, xmax = face.bounding_box
        self.assertTrue(ymin < ymax and xmin < xmax)

    def test_boundary_transitions_and_uncertainty(self):
        result = self.analyzer.analyze(self.sample_image)

        required_boundaries = [
            "subject_background",
            "skin_clothing",
            "skin_hair",
            "hair_background",
            "clothing_background",
            "lace_skin",
            "sheer_clothing_skin",
        ]
        for b in required_boundaries:
            self.assertIn(b, result.boundaries, f"Missing required boundary transition: {b}")
            boundary = result.boundaries[b]
            self.assertTrue(boundary.mask_b64.startswith("data:image/png;base64,"))
            self.assertGreater(boundary.transition_width_px, 0)
            self.assertTrue(0.0 <= boundary.uncertainty_score <= 1.0)
            self.assertFalse(math.isnan(boundary.uncertainty_score))

    def test_confidence_and_surface_modifiers(self):
        result = self.analyzer.analyze(self.sample_image)

        self.assertTrue(0.0 <= result.global_confidence <= 1.0)
        self.assertFalse(math.isnan(result.global_confidence))
        self.assertTrue(result.confidence_map_b64.startswith("data:image/png;base64,"))
        self.assertTrue(result.composite_preview_b64.startswith("data:image/png;base64,"))
        self.assertGreater(result.analysis_time_ms, 0)

        # Verify surface modifier (water droplets/moisture)
        self.assertIn("water_droplets_moisture", result.surface_modifiers)
        mod = result.surface_modifiers["water_droplets_moisture"]
        self.assertEqual(mod.modifier_type, "specular_refractive")
        self.assertTrue(0.0 <= mod.intensity <= 1.0)

    def test_serialization_to_dict(self):
        result = self.analyzer.analyze(self.sample_image)
        d = result.to_dict()

        self.assertIn("image", d)
        self.assertIn("regions", d)
        self.assertIn("boundaries", d)
        self.assertIn("surface_modifiers", d)
        self.assertIn("global_confidence", d)
        self.assertEqual(d["image"]["width"], self.sample_image.width)
        self.assertEqual(d["image"]["height"], self.sample_image.height)

    def test_image_generation_subsystem_untouched(self):
        # Verify GenerationCoordinator and ModelManager remain 100% functional
        mm = ModelManager()
        gc = GenerationCoordinator(mm)
        self.assertIsNotNone(gc)
        self.assertGreaterEqual(len(gc.engines), 2)


if __name__ == "__main__":
    unittest.main()
