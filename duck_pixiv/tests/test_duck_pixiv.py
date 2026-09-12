import unittest
import os
from app.metadata_reader import MetadataReader
from app.tag_generator import TagGenerator
from app.title_generator import TitleGenerator
from app.pixiv_uploader import DryRunUploader
from app.post_manager import PostManager
from app.database import get_all_posts, init_db


class TestDuckPixivAssistant(unittest.TestCase):

    def setUp(self):
        self.sample_image = r"outputs\2026-08-27\Belfast\Belfast_20260827_221641_476836733.png"
        self.abs_sample = os.path.abspath(os.path.join("..", self.sample_image))
        if not os.path.exists(self.abs_sample):
            self.abs_sample = os.path.abspath(self.sample_image)

    def test_metadata_reader(self):
        if not os.path.exists(self.abs_sample):
            self.skipTest("Sample image not found")
        meta = MetadataReader.read_image_metadata(self.abs_sample)
        self.assertIsNotNone(meta)
        self.assertTrue(len(meta["prompt"]) > 0)
        self.assertIn("features", meta)

    def test_tag_generator_controlled_dict(self):
        tag_gen = TagGenerator()
        res = tag_gen.lookup_tag("Belfast")
        self.assertIn("ベルファスト(アズールレーン)", res)

        lingerie_res = tag_gen.lookup_tag("lingerie")
        self.assertIn("ランジェリー", lingerie_res)

        meta = {
            "prompt": "1girl, Belfast, white hair, blue eyes, lingerie, rope bondage, blushing",
            "character_detected": "Belfast"
        }
        tags = tag_gen.generate_tags(meta, max_tags=10)
        self.assertLessEqual(len(tags), 10)
        self.assertIn("ベルファスト(アズールレーン)", tags)
        self.assertIn("AIイラスト", tags)  # From fixed tags

    def test_title_generator(self):
        title_gen = TitleGenerator()
        meta = {
            "character_detected": "Belfast",
            "features": {
                "clothing": ["lingerie", "lace"],
                "expression": ["blushing"],
                "pose": ["kneeling"]
            }
        }
        titles = title_gen.generate_titles(meta, count=5)
        self.assertTrue(len(titles) >= 3)
        self.assertTrue(any("ベルファスト" in t["ja"] for t in titles))
        self.assertTrue(any("en" in t for t in titles))

    def test_dry_run_uploader(self):
        uploader = DryRunUploader()
        if not os.path.exists(self.abs_sample):
            self.skipTest("Sample image not found")
        res = uploader.upload(
            image_paths=[self.abs_sample],
            title="テストタイトル",
            description="テストキャプション",
            tags=["AIイラスト", "ベルファスト"]
        )
        self.assertTrue(res.success)
        self.assertIsNotNone(res.pixiv_id)

    def test_post_manager_full_flow(self):
        if not os.path.exists(self.abs_sample):
            self.skipTest("Sample image not found")
        mgr = PostManager()
        prepared = mgr.prepare_post([self.abs_sample])
        self.assertEqual(len(prepared["title_candidates"]), 5)
        self.assertTrue(len(prepared["tags"]) <= 10)

        # Execute post with explicit DryRun settings for clean testing
        post_res = mgr.execute_post(prepared, settings={"pixiv": {"auth_mode": "dry_run"}})
        self.assertTrue(post_res.success)
        history = get_all_posts(limit=5)
        self.assertTrue(len(history) > 0)


if __name__ == "__main__":
    unittest.main()
