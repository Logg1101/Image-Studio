import unittest
import json
import shutil
import tempfile
from pathlib import Path
from PIL import Image

import config.paths as paths
from core.types import GenerationRequest, GenerationResult, ModelInfo
from StoryStudio.core.catalogue_manager import CatalogueManager
from StoryStudio.core.continuity_tracker import ContinuityTracker
from StoryStudio.core.project_manager import ProjectManager
from StoryStudio.core.pipeline import StoryStudioPipeline
from StoryStudio.agents.llm_provider import LLMProvider
from StoryStudio.agents.prompt_agent.agent import PromptAgent
from StoryStudio.agents.story_agent.agent import StoryAgent

class TestStoryStudio(unittest.TestCase):
    """
    Comprehensive Test Suite for Story Studio Workflow Layer:
    - Data-driven JSON catalogues
    - Character LoRA discovery & metadata parsing
    - Prompt Agent prompt synthesis & Illustrious formatting
    - Story Agent continuity tracking & scene decomposition
    - Project storage persistence (project.json, story.txt, images/, prompts/)
    - Pipeline Single, Batch, and Story Line orchestration
    """

    def setUp(self):
        self.catalogue_mgr = CatalogueManager()
        self.continuity_tracker = ContinuityTracker()
        self.temp_dir = tempfile.mkdtemp(prefix="test_storystudio_")
        self.project_mgr = ProjectManager(projects_root=Path(self.temp_dir) / "projects")
        self.prompt_agent = PromptAgent()
        self.story_agent = StoryAgent()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_catalogues_dynamic_loading(self):
        """Validates that all JSON catalogues exist and load valid data items."""
        catalogues = self.catalogue_mgr.get_all_catalogues()
        self.assertIn("clothes_types", catalogues)
        self.assertIn("clothes_colors", catalogues)
        self.assertIn("clothes_details", catalogues)
        self.assertIn("lighting", catalogues)
        self.assertIn("poses", catalogues)
        self.assertIn("povs", catalogues)
        self.assertIn("bondage_styles", catalogues)
        self.assertIn("gag_types", catalogues)
        self.assertIn("expressions", catalogues)
        self.assertIn("scenes", catalogues)
        self.assertIn("styles", catalogues)
        self.assertIn("resolutions", catalogues)
        self.assertIn("upscalers", catalogues)
        self.assertIn("samplers", catalogues)
        self.assertIn("schedulers", catalogues)

        # Check 16 expressions
        expressions = catalogues["expressions"]
        self.assertGreaterEqual(len(expressions), 15)
        e_ids = [e["id"] for e in expressions]
        self.assertIn("neutral", e_ids)
        self.assertIn("sweet_smile", e_ids)
        self.assertIn("shy_blush", e_ids)
        self.assertIn("seductive", e_ids)
        self.assertIn("ahegao_ecstatic", e_ids)

        # Check upscalers
        upscalers = catalogues["upscalers"]
        self.assertIn("4x-UltraSharp", upscalers)
        self.assertIn("None", upscalers)

        # Check resolutions
        resolutions = catalogues["resolutions"]
        self.assertGreaterEqual(len(resolutions), 12)

        # Check 30 clothes types
        clothes_types = catalogues["clothes_types"]
        self.assertGreaterEqual(len(clothes_types), 30)
        c_ids = [c["id"] for c in clothes_types]
        self.assertIn("maid", c_ids)
        self.assertIn("school_uniform", c_ids)
        self.assertIn("bunny_girl", c_ids)
        self.assertIn("cheongsam_qipao", c_ids)
        self.assertIn("cyberpunk_techwear", c_ids)
        self.assertIn("leather_catsuit", c_ids)

        # Check multi-select details & accessories
        details = catalogues["clothes_details"]
        self.assertGreaterEqual(len(details), 20)
        d_ids = [d["id"] for d in details]
        self.assertIn("lace", d_ids)
        self.assertIn("long_gloves", d_ids)
        self.assertIn("garter_belt", d_ids)
        self.assertIn("fishnets", d_ids)
        self.assertIn("bell_collar", d_ids)

        # Check 10 bondage styles
        bondages = catalogues["bondage_styles"]
        self.assertGreaterEqual(len(bondages), 10)
        b_ids = [b["id"] for b in bondages]
        self.assertIn("classic_shibari", b_ids)
        self.assertIn("suspended_bondage", b_ids)

        # Check 10 gag types
        gags = catalogues["gag_types"]
        self.assertGreaterEqual(len(gags), 10)
        g_ids = [g["id"] for g in gags]
        self.assertIn("ball_gag", g_ids)
        self.assertIn("tape_gag", g_ids)

        # Check scenes & styles
        scenes = catalogues["scenes"]
        self.assertGreaterEqual(len(scenes), 10)
        s_ids = [s["id"] for s in scenes]
        self.assertIn("first_meeting", s_ids)
        self.assertIn("getting_ready", s_ids)

    def test_02_character_lora_discovery(self):
        """Validates dynamic character discovery and metadata extraction from models/loras."""
        characters = self.catalogue_mgr.scan_characters()
        self.assertGreater(len(characters), 0)

        # Check Belfast (with metadata)
        belfast = next((c for c in characters if c["id"] == "Belfast_HassakuXIllustrious_V1"), None)
        self.assertIsNotNone(belfast)
        self.assertEqual(belfast["name"], "Belfast (Royal Maid)")
        self.assertIn("belfast", belfast["trigger"].lower())
        self.assertEqual(belfast["default_weight"], 0.85)

        # Check Rias (with new hardcoded prompt)
        rias = next((c for c in characters if "rias" in c["id"].lower()), None)
        self.assertIsNotNone(rias)
        self.assertEqual(rias["name"], "Rias Gremory")
        self.assertIn("red hair", rias["trigger"])
        self.assertIn("large breasts", rias["trigger"])

        # Check all 6 user-requested hardcoded character triggers
        expected_hardcoded = {
            "belfast": "Belfast, long white hair, blue eyes, large breasts",
            "akeno": "Akeno Himejima, Long black hair, purple eyes, large breasts",
            "rias": "Rias Gremory, long red hair, green eyes, large breasts",
            "taihou": "Taihou, long black hair, red eyes, large breasts",
            "velina": "Velina Airgid, long white hair, purple eyes, large breasts",
            "rita": "Rita, short brown hair, red eyes, large breasts"
        }
        for k, expected_trig in expected_hardcoded.items():
            char = next((c for c in characters if k in c["id"].lower()), None)
            if char:
                self.assertEqual(char["trigger"], expected_trig)

    def test_03_prompt_agent_synthesis(self):
        """Validates Prompt Agent assembling tags in strict Illustrious format including lighting, bondage, and gags."""
        params = {
            "character_name": "Silver Maid",
            "character_trigger": "silvermaid, silver hair, violet eyes",
            "clothes_type": "maid",
            "clothes_type_prompt": "maid dress, frilled apron",
            "clothes_color": "white",
            "clothes_details": ["delicate lace trim", "opera gloves", "silk ribbons"],
            "lighting": "dim_candlelight",
            "lighting_prompt": "warm candlelight, cozy intimate ambiance",
            "pose": "looking into mirror",
            "pose_prompt": "looking into full-length mirror",
            "pov": "three_quarter_front",
            "pov_prompt": "three-quarter front view",
            "bondage_style": "classic_shibari",
            "bondage_prompt": "intricate shibari rope bondage",
            "gag_type": "ball_gag",
            "gag_prompt": "red silicone ball gag",
            "scene": "getting_ready",
            "scene_description": "dressing room, ornate vanity, soft morning light",
            "style": "romantic_anime",
            "style_prompt": "romantic anime art style, soft pastel palette"
        }

        result = self.prompt_agent.generate_prompts(params)
        self.assertIn("positive_prompt", result)
        self.assertIn("negative_prompt", result)

        pos = result["positive_prompt"]
        # Must contain masterpiece declaration
        self.assertIn("masterpiece", pos)
        self.assertIn("1girl", pos)
        # Must contain trigger
        self.assertIn("silvermaid", pos)
        # Must contain clothing
        self.assertIn("white", pos)
        self.assertIn("delicate lace trim", pos)
        self.assertIn("opera gloves", pos)
        # Must contain bondage & gag
        self.assertIn("intricate shibari rope bondage", pos)
        self.assertIn("red silicone ball gag", pos)
        # Must contain pose & pov
        self.assertIn("looking into full-length mirror", pos)
        self.assertIn("three-quarter front view", pos)
        # Must end with fixed aesthetic suffix
        self.assertTrue(pos.endswith("realistic fabric texture, realistic lighting, cinematic"))
        # Must contain negative prompt
        neg = result["negative_prompt"]
        self.assertIn("bad anatomy", neg)
        self.assertIn("extra fingers", neg)

    def test_04_story_agent_continuity_tracking(self):
        """Validates Story Agent multi-line continuity progression."""
        story_lines = [
            "She arrives at the grand mansion and nervously looks around.",
            "She enters her new private bedroom.",
            "She examines the beautiful outfit waiting on the bed.",
            "She changes into the outfit and looks at herself in the mirror.",
            "She hears someone approaching the room."
        ]

        char_info = {
            "id": "silver_maid",
            "name": "Silver Maid",
            "trigger": "silvermaid, silver hair, violet eyes",
            "description": "adult silver-haired maid"
        }
        defaults = {
            "clothes_type": "maid",
            "clothes_color": "white",
            "clothes_details": ["lace", "ribbons"],
            "style": "soft_romantic_anime"
        }

        continuity = {}

        # Line 1: Arrival
        d1 = self.story_agent.interpret_story_line(
            current_line=story_lines[0],
            line_index=0,
            full_story=story_lines,
            continuity_state=continuity,
            character_info=char_info,
            default_settings=defaults
        )
        self.tracker = ContinuityTracker()
        self.tracker.update_from_directive(d1)
        c1 = self.tracker.get_state()
        self.assertIn("mansion", c1["location"].lower())

        # Line 2: Bedroom entry
        d2 = self.story_agent.interpret_story_line(
            current_line=story_lines[1],
            line_index=1,
            full_story=story_lines,
            continuity_state=c1,
            character_info=char_info,
            default_settings=defaults
        )
        self.tracker.update_from_directive(d2)
        c2 = self.tracker.get_state()
        self.assertIn("bedroom", c2["location"].lower())

        # Line 3: Examines outfit (still in bedroom)
        d3 = self.story_agent.interpret_story_line(
            current_line=story_lines[2],
            line_index=2,
            full_story=story_lines,
            continuity_state=c2,
            character_info=char_info,
            default_settings=defaults
        )
        self.tracker.update_from_directive(d3)
        c3 = self.tracker.get_state()
        self.assertIn("bedroom", c3["location"].lower())

        # Line 4: Mirror
        d4 = self.story_agent.interpret_story_line(
            current_line=story_lines[3],
            line_index=3,
            full_story=story_lines,
            continuity_state=c3,
            character_info=char_info,
            default_settings=defaults
        )
        self.tracker.update_from_directive(d4)
        c4 = self.tracker.get_state()
        self.assertIn("mirror", c4["pose"].lower())

    def test_05_project_storage_persistence(self):
        """Validates project disk persistence under StoryStudio/projects/<ProjectName>/."""
        proj = self.project_mgr.create_or_update_project(
            name="TestMansionStory",
            story_lines=["Line 1 text", "Line 2 text"],
            character_id="Belfast_HassakuXIllustrious_V1",
            default_settings={"clothes_type": "maid", "clothes_color": "white"}
        )

        p_dir = self.project_mgr.get_project_dir("TestMansionStory")
        self.assertTrue((p_dir / "project.json").exists())
        self.assertTrue((p_dir / "story.txt").exists())
        self.assertTrue((p_dir / "images").exists())
        self.assertTrue((p_dir / "prompts").exists())

        # Save dummy generated image for line 0
        dummy_img = Image.new("RGB", (64, 64), color="purple")
        prompt_meta = {
            "line_index": 0,
            "line_text": "Line 1 text",
            "positive_prompt": "masterpiece, 1girl, belfast",
            "negative_prompt": "bad anatomy",
            "seed": 424242
        }

        saved_path = self.project_mgr.save_line_result(
            project_name="TestMansionStory",
            line_index=0,
            image=dummy_img,
            prompt_meta=prompt_meta,
            continuity_snapshot={"location": "mansion"}
        )

        self.assertTrue(Path(saved_path).exists())
        self.assertTrue((p_dir / "images" / "001.png").exists())
        self.assertTrue((p_dir / "prompts" / "001.json").exists())

        # Check updated project.json
        updated_proj = self.project_mgr.get_project("TestMansionStory")
        self.assertEqual(updated_proj["lines"][0]["status"], "Completed")
        self.assertEqual(updated_proj["lines"][0]["seed"], 424242)

    def test_06_pipeline_orchestration_mock(self):
        """Validates pipeline execution using mock coordinator."""
        class MockCoordinator:
            def generate(self, req: GenerationRequest) -> GenerationResult:
                # Create a temporary dummy output image
                out_path = Path(tempfile.gettempdir()) / f"mock_gen_{req.seed}.png"
                img = Image.new("RGB", (req.width, req.height), color="pink")
                img.save(out_path)
                meta_path = out_path.with_suffix(".json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump({"prompt": req.prompt}, f)

                return GenerationResult(
                    image_path=str(out_path),
                    metadata_path=str(meta_path),
                    generation_time_ms=120.0,
                    vram_peak_gb=4.5
                )

        mock_coord = MockCoordinator()
        pipe = StoryStudioPipeline(
            coordinator=mock_coord,
            catalogue_manager=self.catalogue_mgr,
            project_manager=self.project_mgr,
            prompt_agent=self.prompt_agent,
            story_agent=self.story_agent
        )

        # 1. Single Generation
        single_res = pipe.generate_single({
            "character_id": "Belfast_HassakuXIllustrious_V1",
            "clothes_type": "maid",
            "clothes_color": "white",
            "clothes_details": ["lace"],
            "pose": "standing",
            "pov": "three_quarter_front",
            "scene": "first_meeting",
            "style": "romantic_anime",
            "width": 1024,
            "height": 1536,
            "seed": 99999
        })
        self.assertIn("image_url", single_res)
        self.assertTrue(single_res["image_url"].startswith("data:image/png;base64,"))
        self.assertEqual(single_res["seed"], 99999)

        # 2. Batch Generation (3 images)
        batch_results = pipe.generate_batch({
            "character_id": "Belfast_HassakuXIllustrious_V1",
            "clothes_type": "maid",
            "clothes_color": "white",
            "clothes_details": ["lace"],
            "pose": "standing",
            "pov": "three_quarter_front",
            "scene": "first_meeting",
            "style": "romantic_anime",
            "width": 1024,
            "height": 1536,
            "seed": -1
        }, batch_count=3)
        self.assertEqual(len(batch_results), 3)

        # 3. Story Line Execution
        self.project_mgr.create_or_update_project(
            name="EpisodicMansion",
            story_lines=["Line 1: Belfast welcomes the master.", "Line 2: Belfast serves afternoon tea."],
            character_id="Belfast_HassakuXIllustrious_V1",
            default_settings={"width": 1024, "height": 1536, "steps": 30}
        )

        line_res = pipe.generate_story_line("EpisodicMansion", 0)
        self.assertEqual(line_res["status"], "Completed")
        self.assertIn("belfast", line_res["positive_prompt"].lower())

        # Check line 1 is updated in project.json
        proj = self.project_mgr.get_project("EpisodicMansion")
        self.assertEqual(proj["lines"][0]["status"], "Completed")
        self.assertEqual(proj["lines"][1]["status"], "Pending")

    def test_07_decompose_story_and_prompt_override(self):
        """Validates breaking down a single story box into scenes, editing prompts, and generating."""
        class MockCoordinator:
            def __init__(self):
                self.last_prompt = None
            def generate(self, req: GenerationRequest) -> GenerationResult:
                self.last_prompt = req.prompt
                out_path = Path(tempfile.gettempdir()) / f"mock_story_{req.seed}.png"
                img = Image.new("RGB", (req.width, req.height), color="violet")
                img.save(out_path)
                meta_path = out_path.with_suffix(".json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump({"prompt": req.prompt}, f)

                return GenerationResult(
                    image_path=str(out_path),
                    metadata_path=str(meta_path),
                    generation_time_ms=100.0,
                    vram_peak_gb=4.0
                )

        mock_coord = MockCoordinator()
        pipe = StoryStudioPipeline(
            coordinator=mock_coord,
            catalogue_manager=self.catalogue_mgr,
            project_manager=self.project_mgr,
            prompt_agent=self.prompt_agent,
            story_agent=self.story_agent
        )

        full_raw_story = (
            "She arrives at the grand palace and looks around in wonder. "
            "She walks into the royal ballroom where crystal chandeliers glow. "
            "She is greeted by the head butler who bows courteously."
        )

        # 1. Decompose story into scenes
        scenes = pipe.decompose_story_to_scenes(
            story_text=full_raw_story,
            character_id="Belfast_HassakuXIllustrious_V1",
            default_settings={"clothes_type": "evening_dress", "clothes_color": "blue"}
        )

        self.assertGreaterEqual(len(scenes), 3)
        self.assertEqual(scenes[0]["index"], 0)
        self.assertIn("palace", scenes[0]["text"].lower())
        self.assertIn("masterpiece", scenes[0]["positive_prompt"])
        self.assertIn("belfast", scenes[0]["positive_prompt"].lower())

        # 2. User edits the prompt for scene 1 (ballroom)
        edited_custom_prompt = scenes[1]["positive_prompt"] + ", customized glowing fairy lights, user edited tag"
        scenes[1]["positive_prompt"] = edited_custom_prompt

        # 3. Create project with edited scenes
        self.project_mgr.create_or_update_project(
            name="CustomBallroomStory",
            character_id="Belfast_HassakuXIllustrious_V1",
            scenes_data=scenes,
            raw_story_text=full_raw_story
        )

        # 4. Generate scene 1
        res = pipe.generate_story_line("CustomBallroomStory", 1)
        self.assertEqual(res["status"], "Completed")
        self.assertEqual(res["positive_prompt"], edited_custom_prompt)
        # Verify coordinator received the exact user-edited prompt
        self.assertEqual(mock_coord.last_prompt, edited_custom_prompt)

if __name__ == "__main__":
    unittest.main()
