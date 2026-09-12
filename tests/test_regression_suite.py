import unittest
import json
import gc
import time
import torch
from pathlib import Path
from PIL import Image

from core.types import GenerationRequest, ModelInfo, MAX_SEED, sanitize_seed
from core.diagnostics.system_check import SystemDiagnostics
from core.project_manager import ProjectManager
from core.usage_tracker import UsageTracker
from adapters.base import AdapterType, AdapterStatus
from adapters.lora.detector import LoRAFormatDetector
from adapters.lora.inspector import LoRACheckpointInspector
from adapters.lora.converter import LoRAKeyConverter
from adapters.registry import adapter_registry
from ui.state import model_manager
from engines.sdxl.engine import SDXLEngine

class TestImageStudioFullSuite(unittest.TestCase):

    def test_01_generation_request_schema(self):
        dummy_model = ModelInfo("test", "sdxl", "base", "safetensors", "")
        req = GenerationRequest(
            model=dummy_model,
            prompt="1girl, solo, belfast",
            pulid_enabled=True,
            pulid_strength=0.85,
            project_name="Belfast Set",
            character_name="Belfast"
        )
        self.assertEqual(req.pulid_strength, 0.85)
        self.assertEqual(req.pulid_scale, 0.85)
        self.assertEqual(req.project_name, "Belfast Set")
        self.assertEqual(req.character_name, "Belfast")
        self.assertEqual(req.width, 1024)
        self.assertEqual(req.height, 1024)

    def test_02_seed_boundary_and_overflow_prevention(self):
        self.assertEqual(sanitize_seed(0), 0)
        self.assertEqual(sanitize_seed(2147483647), 2147483647)
        oversized = 2764734946
        sanitized = sanitize_seed(oversized)
        self.assertLessEqual(sanitized, MAX_SEED)
        self.assertGreaterEqual(sanitized, 0)
        self.assertEqual(sanitize_seed(-12345), 12345)
        self.assertEqual(sanitize_seed("987654"), 987654)

    def test_03_parameter_serialization_and_reuse(self):
        dummy_model = ModelInfo("test_model", "sdxl", "base", "safetensors", "")
        req = GenerationRequest(
            model=dummy_model,
            prompt="masterpiece test",
            seed=3500000000,
            pulid_strength=0.9,
            project_name="DemoProj",
            character_name="Akeno"
        )
        serialized = json.dumps({
            "model": req.model.id,
            "prompt": req.prompt,
            "seed": req.seed,
            "pulid_strength": req.pulid_strength,
            "project": req.project_name,
            "character": req.character_name
        })
        deserialized = json.loads(serialized)
        reused_seed = sanitize_seed(deserialized.get("seed"))
        self.assertLessEqual(reused_seed, MAX_SEED)
        self.assertEqual(reused_seed, req.seed)
        self.assertEqual(deserialized.get("project"), "DemoProj")

    def test_03b_history_manager_metadata_preserves_model(self):
        from history.manager import HistoryManager
        from core.types import GenerationResult
        import tempfile

        dummy_model = ModelInfo("Illustrious-XL-v1.0", "sdxl", "base", "safetensors", "")
        req = GenerationRequest(
            model=dummy_model,
            prompt="test prompt for metadata preservation",
            seed=42,
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            temp_img_path = tf.name

        meta_json_path = ""
        try:
            res = GenerationResult(
                image_path=temp_img_path,
                metadata_path="",
                generation_time_ms=150.0,
                vram_peak_gb=4.5,
            )
            hm = HistoryManager()
            meta_json_path = hm.save_generation(req, res)
            self.assertTrue(Path(meta_json_path).exists())

            with open(meta_json_path, "r", encoding="utf-8") as f:
                saved_meta = json.load(f)

            self.assertIn("model", saved_meta)
            self.assertEqual(saved_meta["model"], "Illustrious-XL-v1.0")
        finally:
            if Path(temp_img_path).exists():
                Path(temp_img_path).unlink()
            if meta_json_path and Path(meta_json_path).exists():
                Path(meta_json_path).unlink()

    def test_04_project_manager_unique_naming(self):
        ProjectManager.ensure_directories()
        
        # Test default hierarchy
        p1 = ProjectManager.get_output_destination(character="Belfast", project="Personal", seed=1001)
        self.assertIn("Belfast", p1.name)
        self.assertIn("1001", p1.name)
        self.assertTrue(str(p1).endswith(".png"))

        # Test project hierarchy
        p2 = ProjectManager.get_output_destination(character="Belfast", project="AzurLaneProject", seed=1002)
        self.assertIn("AzurLaneProject", str(p2))
        self.assertIn("Belfast", p2.name)

        # Test collision avoidance
        p1.write_text("test")
        p1_collision = ProjectManager.get_output_destination(character="Belfast", project="Personal", seed=1001)
        self.assertNotEqual(str(p1), str(p1_collision))
        self.assertIn("_01.png", str(p1_collision))
        p1.unlink()

    def test_05_character_detection(self):
        self.assertEqual(ProjectManager.detect_character("1girl, belfast, azure lane, maid"), "Belfast")
        self.assertEqual(ProjectManager.detect_character("akeno himejima, highschool dxd"), "Akeno")
        self.assertEqual(ProjectManager.detect_character("masterpiece, best quality, 1girl, solo"), "General")

    def test_06_usage_tracker(self):
        UsageTracker.log_generation(
            checkpoint="hassakuXLIllustrious_v22",
            sampler="Euler a",
            loras_count=1,
            is_img2img=False,
            gen_time_sec=12.5,
            vram_peak_gb=10.2
        )
        summary = UsageTracker.get_summary()
        self.assertGreater(summary["today"]["generated"], 0)
        self.assertGreater(summary["today"]["peak_vram_gb"], 0.0)
        self.assertIn("hassakuXLIllustrious_v22", summary["all_time"]["checkpoints"])

    def test_07_universal_lora_belfast_detection(self):
        belfast = Path("models/loras/characters/Belfast_HassakuXL.safetensors")
        if not belfast.exists():
            self.skipTest("Belfast_HassakuXL not present.")
        info = LoRAFormatDetector.detect(belfast)
        self.assertEqual(info.name, "Belfast_HassakuXL")
        self.assertEqual(info.architecture, "sdxl")
        self.assertEqual(info.adapter_type, AdapterType.LYCORIS)
        self.assertEqual(info.tensor_count, 2382)

    def test_08_lora_key_conversion(self):
        fake_state_dict = {
            "lora_unet_add_embedding_linear_1.lora_down.weight": torch.randn(32, 2816),
            "lora_unet_conv_in.lora_down.weight": torch.randn(32, 4),
            "lycoris_mid_block_attentions_0_to_q.lora_down.weight": torch.randn(32, 640),
        }
        normalized, stats = LoRAKeyConverter.normalize_state_dict(fake_state_dict, target_architecture="sdxl")
        self.assertIn("lora_unet_conv_in.lora_down.weight", normalized)
        self.assertIn("lora_unet_mid_block_attentions_0_to_q.lora_down.weight", normalized)
        self.assertNotIn("lora_unet_add_embedding_linear_1.lora_down.weight", normalized)

    def test_09_adapter_discovery(self):
        loras = adapter_registry.scan_loras()
        cnets = adapter_registry.scan_controlnets()
        ips = adapter_registry.scan_ip_adapters()
        self.assertIsInstance(loras, dict)
        self.assertIsInstance(cnets, dict)
        self.assertIsInstance(ips, dict)

    def test_10_system_diagnostics_report(self):
        report = SystemDiagnostics.generate_report()
        self.assertIn("IMAGESTUDIO WORKSTATION SYSTEM STATUS", report)
        self.assertIn("Triton: Optional", report)

    def test_11_end_to_end_generation_and_memory_stability(self):
        model_manager.scan_models()
        models = [m for m in model_manager.available_models.values() if m.architecture.lower() == "sdxl"]
        if not models:
            self.skipTest("No SDXL models found.")

        engine = SDXLEngine()
        engine.load(models[0])

        # 1. Base txt2img with ProjectManager destination
        req_base = GenerationRequest(
            model=models[0],
            prompt="1girl, solo, masterpiece",
            width=512,
            height=512,
            steps=3,
            seed=100,
            project_name="TestProject",
            character_name="SoloGirl"
        )
        res_base = engine.generate(req_base)
        self.assertTrue(Path(res_base.image_path).exists())
        self.assertIn("SoloGirl", Path(res_base.image_path).name)

        # 2. Belfast LoRA
        belfast = Path("models/loras/characters/Belfast_HassakuXL.safetensors")
        if belfast.exists():
            req_lora = GenerationRequest(
                model=models[0],
                prompt="1girl, solo, belfast, masterpiece",
                width=512,
                height=512,
                steps=3,
                seed=101,
                loras={str(belfast.resolve()): 1.0},
                character_name="Belfast"
            )
            res_lora = engine.generate(req_lora)
            self.assertTrue(Path(res_lora.image_path).exists())

        # 3. LoRA Detachment and Baseline recovery
        req_clean = GenerationRequest(
            model=models[0],
            prompt="1girl, solo, masterpiece",
            width=512,
            height=512,
            steps=3,
            seed=102,
            loras={}
        )
        res_clean = engine.generate(req_clean)
        self.assertTrue(Path(res_clean.image_path).exists())

        # 4. 10-run memory drift check
        allocs = []
        for i in range(10):
            req_loop = GenerationRequest(
                model=models[0],
                prompt=f"loop scene {i}, 1girl",
                width=512,
                height=512,
                steps=2,
                seed=200 + i
            )
            engine.generate(req_loop)
            gc.collect()
            if torch.cuda.is_available():
                allocs.append(torch.cuda.memory_allocated(0) / (1024 ** 2))

        if allocs:
            drift = allocs[-1] - allocs[0]
            self.assertLess(abs(drift), 1.0, f"Memory drift detected: {drift} MB")

        engine.unload()

    def test_12_history_db_delete_record_cleanup(self):
        from history.database import HistoryDB
        import config.paths as paths

        db = HistoryDB()
        dummy_img = paths.IMAGES_DIR / "test_del_reg.png"
        dummy_meta = paths.METADATA_DIR / "test_del_reg.json"
        dummy_img.parent.mkdir(parents=True, exist_ok=True)
        dummy_meta.parent.mkdir(parents=True, exist_ok=True)
        dummy_img.write_bytes(b"test")
        dummy_meta.write_text("{}", encoding="utf-8")

        rec_id = db.insert_record({
            "model_id": "test_m",
            "architecture": "sdxl",
            "variant": "base",
            "prompt": "test",
            "negative_prompt": "",
            "seed": 42,
            "steps": 1,
            "guidance_scale": 7.0,
            "width": 512,
            "height": 512,
            "image_path": str(dummy_img.resolve()),
            "generation_time_ms": 100.0,
            "vram_peak_gb": 1.0,
        })
        self.assertTrue(dummy_img.exists())
        self.assertTrue(dummy_meta.exists())

        res = db.delete_record(rec_id, delete_file=True)
        self.assertTrue(res)
        self.assertFalse(dummy_img.exists())
        self.assertFalse(dummy_meta.exists())

    def test_13_history_db_delete_record_logs_oserror(self):
        from history.database import HistoryDB
        import config.paths as paths
        from unittest.mock import patch

        db = HistoryDB()
        dummy_img = paths.IMAGES_DIR / "test_err_reg.png"
        dummy_meta = paths.METADATA_DIR / "test_err_reg.json"
        dummy_img.parent.mkdir(parents=True, exist_ok=True)
        dummy_meta.parent.mkdir(parents=True, exist_ok=True)
        dummy_img.write_bytes(b"test")
        dummy_meta.write_text("{}", encoding="utf-8")

        rec_id = db.insert_record({
            "model_id": "test_m",
            "architecture": "sdxl",
            "variant": "base",
            "prompt": "test",
            "negative_prompt": "",
            "seed": 43,
            "steps": 1,
            "guidance_scale": 7.0,
            "width": 512,
            "height": 512,
            "image_path": str(dummy_img.resolve()),
            "generation_time_ms": 100.0,
            "vram_peak_gb": 1.0,
        })

        with patch.object(Path, "unlink", side_effect=[None, OSError("Lock error")]):
            with patch("builtins.print") as mock_print:
                res = db.delete_record(rec_id, delete_file=True)
                self.assertTrue(res)
                mock_print.assert_any_call(f"[HistoryDB] Could not delete metadata file {dummy_meta}: Lock error")

        if dummy_img.exists():
            dummy_img.unlink()
        if dummy_meta.exists():
            dummy_meta.unlink()

    def test_14_vendored_supir_names_defined(self):
        """Vendored SUPIR/LLaVA modules must define dist and checkpoint_wrapper to prevent F821."""
        import ast

        files_to_check = [
            ("engines/SUPIR/llava/model/language_model/mpt/modeling_mpt.py", "dist"),
            ("engines/SUPIR/SUPIR/modules/SUPIR_v0.py", "checkpoint_wrapper"),
            ("engines/SUPIR/sgm/modules/diffusionmodules/openaimodel.py", "checkpoint_wrapper"),
        ]
        for rel_path, symbol in files_to_check:
            tree = ast.parse(Path(rel_path).read_text(encoding="utf-8"))
            defined = False
            for node in ast.walk(tree):
                if isinstance(node, (ast.Name, ast.alias)) and (
                    getattr(node, "id", None) == symbol
                    or getattr(node, "name", None) == symbol
                    or getattr(node, "asname", None) == symbol
                ):
                    defined = True
                    break
            self.assertTrue(defined, f"{symbol} should be defined in {rel_path}")


    def test_15_async_route_handlers_no_blocking_calls(self):
        """Async FastAPI handlers in StoryStudio and duck_pixiv must not make blocking network/file calls."""
        import subprocess
        import sys

        # Verify ruff ASYNC checks pass cleanly
        cmd = [sys.executable, "-m", "ruff", "check", "--select", "ASYNC", "StoryStudio/router.py", "duck_pixiv/ui/web_app.py"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Ruff ASYNC check failed:\n{res.stdout}\n{res.stderr}")

        # Check StoryStudio/router.py wraps urlopen inside threadpool
        router_src = Path("StoryStudio/router.py").read_text(encoding="utf-8")
        self.assertIn("run_in_threadpool", router_src)
        self.assertIn("data_url.startswith((\"http://\", \"https://\"))", router_src)

        # Check duck_pixiv/ui/web_app.py uses FileResponse for index
        web_app_src = Path("duck_pixiv/ui/web_app.py").read_text(encoding="utf-8")
        self.assertIn("def serve_index():", web_app_src)
        self.assertIn("return FileResponse(index_file, media_type=\"text/html\")", web_app_src)


    def test_16_model_detector_gguf_and_deep_safetensors(self):
        """ModelArchitectureDetector must correctly inspect GGUF and safetensors with keys beyond index 50."""
        from core.models.detector import ModelArchitectureDetector
        import safetensors.torch
        import tempfile

        # 1. GGUF inspection
        flux_gguf = Path("models/checkpoints/Flux/flux1-schnell-q4.gguf")
        if flux_gguf.exists():
            res = ModelArchitectureDetector.detect(flux_gguf)
            self.assertEqual(res["architecture"], "flux")
            self.assertEqual(res["variant"], "schnell")
            self.assertEqual(res["format"], "gguf")

        dev_gguf = Path("models/checkpoints/Flux/modern_anime_full_Q4_0.gguf")
        if dev_gguf.exists():
            res = ModelArchitectureDetector.detect(dev_gguf)
            self.assertEqual(res["architecture"], "flux")
            self.assertEqual(res["variant"], "dev")
            self.assertEqual(res["format"], "gguf")

        # 2. Safetensors key beyond index 50
        tmp = Path(tempfile.gettempdir()) / "test_reg_late_key.safetensors"
        try:
            d = {f"prefix.layer.{i}.bias": torch.zeros((2, 2)) for i in range(70)}
            d["double_blocks.0.img_attn.proj.weight"] = torch.zeros((2, 2))
            safetensors.torch.save_file(d, str(tmp))
            res = ModelArchitectureDetector.detect(tmp)
            self.assertEqual(res["architecture"], "flux")
            self.assertEqual(res["variant"], "schnell")
            self.assertEqual(res["format"], "safetensors")
        finally:
            if tmp.exists():
                tmp.unlink()


    def test_17_flux_engine_lora_unload_and_invalidation(self):
        """FluxEngine must log warnings and clear pipeline reference if unload_lora_weights fails."""
        import tempfile
        from unittest.mock import MagicMock, patch
        from engines.flux.engine import FluxEngine

        engine = FluxEngine()
        model_info = ModelInfo("flux_test", "flux", "schnell", "gguf", "dummy_path.gguf")

        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.images = [Image.new("RGB", (64, 64), color="green")]
        mock_pipe.return_value = mock_output
        mock_pipe.unload_lora_weights.side_effect = RuntimeError("GPU memory access error detaching LoRA")

        engine.pipeline = mock_pipe
        engine.current_model_info = model_info

        req = GenerationRequest(
            model=model_info,
            prompt="1girl in meadow",
            width=64,
            height=64,
            steps=1,
            seed=123,
            loras={"dummy_adapter.safetensors": 0.75}
        )

        out_tmp = str(Path(tempfile.gettempdir()) / "test_reg_flux_out.png")
        with patch("core.project_manager.ProjectManager.get_output_destination", return_value=out_tmp), \
             patch("core.usage_tracker.UsageTracker.log_generation"):
            res = engine.generate(req)

        mock_pipe.unload_lora_weights.assert_called_once()
        # Verify pipeline reference was cleanly invalidated to avoid bleed-through
        self.assertIsNone(engine.pipeline)
        self.assertIsNone(engine.current_model_info)


if __name__ == "__main__":
    unittest.main()
