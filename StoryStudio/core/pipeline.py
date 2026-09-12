import io
import re
import base64
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from PIL import Image

import config.paths as paths
from core.types import GenerationRequest, GenerationResult, ModelInfo, sanitize_seed, MAX_SEED
from core.model_manager import ModelManager
from core.generation import GenerationCoordinator
from StoryStudio.agents.prompt_agent.agent import PromptAgent
from StoryStudio.agents.story_agent.agent import StoryAgent
from StoryStudio.core.catalogue_manager import CatalogueManager
from StoryStudio.core.continuity_tracker import ContinuityTracker
from StoryStudio.core.project_manager import ProjectManager

class StoryStudioPipeline:
    """
    Story Studio Orchestration Pipeline.
    Sits strictly ABOVE the existing ImageStudio Generation Engine and coordinates:
    - Prompt Agent (Prompt Synthesis)
    - Story Agent (Scene Interpretation & Continuity)
    - Hassaku XL Illustrious SDXL Generation
    - Single, Batch, and Story line-by-line execution
    """

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        coordinator: Optional[GenerationCoordinator] = None,
        prompt_agent: Optional[PromptAgent] = None,
        story_agent: Optional[StoryAgent] = None,
        catalogue_manager: Optional[CatalogueManager] = None,
        project_manager: Optional[ProjectManager] = None
    ):
        self.model_manager = model_manager or ModelManager()
        self.coordinator = coordinator or GenerationCoordinator(self.model_manager)
        self.prompt_agent = prompt_agent or PromptAgent()
        self.story_agent = story_agent or StoryAgent()
        self.catalogue_manager = catalogue_manager or CatalogueManager()
        self.project_manager = project_manager or ProjectManager()
        self.continuity_tracker = ContinuityTracker()
        self.is_cancelled = False

    def get_primary_model_info(self) -> ModelInfo:
        """
        Locates the primary Hassaku XL Illustrious SDXL checkpoint.
        Falls back to any registered SDXL model if specific name differs.
        """
        models = self.model_manager.available_models
        # 1. Look specifically for Hassaku XL Illustrious
        for k, m in models.items():
            if "hassaku" in k.lower() and m.architecture == "sdxl":
                return m

        # 2. Look for any SDXL model
        for k, m in models.items():
            if m.architecture == "sdxl":
                return m

        # 3. Fallback to any available model
        if models:
            return next(iter(models.values()))

        # 4. Construct placeholder if not yet scanned
        sdxl_file = paths.SDXL_MODELS_DIR / "hassakuXLIllustrious_v22.safetensors"
        return ModelInfo(
            id="hassakuXLIllustrious_v22",
            architecture="sdxl",
            variant="base",
            format="safetensors",
            transformer_path=str(sdxl_file.resolve())
        )

    def prepare_single_prompt(self, params: Dict[str, Any]) -> Dict[str, str]:
        """
        Queries the Prompt Agent with user selections and returns { positive_prompt, negative_prompt }.
        Resolves prompt descriptions from catalogues when item IDs are supplied.
        """
        cats = self.catalogue_manager.get_all_catalogues()
        agent_params = dict(params)

        # Helper to lookup catalogue item prompt by id
        def lookup_cat(cat_key: str, val: Any) -> str:
            if not val:
                return ""
            val_str = str(val).strip()
            for item in cats.get(cat_key, []):
                if item.get("id") == val_str:
                    return item.get("prompt") or item.get("name") or val_str
            return val_str

        if "expression" in agent_params and not agent_params.get("expression_prompt"):
            agent_params["expression_prompt"] = lookup_cat("expressions", agent_params["expression"])

        if "clothes_type" in agent_params and not agent_params.get("clothes_type_prompt"):
            agent_params["clothes_type_prompt"] = lookup_cat("clothes_types", agent_params["clothes_type"])

        if "lighting" in agent_params and not agent_params.get("lighting_prompt"):
            agent_params["lighting_prompt"] = lookup_cat("lighting", agent_params["lighting"])

        if "pose" in agent_params and not agent_params.get("pose_prompt"):
            agent_params["pose_prompt"] = lookup_cat("poses", agent_params["pose"])

        if "pov" in agent_params and not agent_params.get("pov_prompt"):
            agent_params["pov_prompt"] = lookup_cat("povs", agent_params["pov"])

        if "bondage_type" in agent_params and not agent_params.get("bondage_type_prompt"):
            agent_params["bondage_type_prompt"] = lookup_cat("bondage_types", agent_params["bondage_type"])

        if "bondage_style" in agent_params and not agent_params.get("bondage_style_prompt"):
            agent_params["bondage_style_prompt"] = lookup_cat("bondage_styles", agent_params["bondage_style"])
        elif "bondage" in agent_params and not agent_params.get("bondage_style_prompt"):
            agent_params["bondage_style_prompt"] = lookup_cat("bondage_styles", agent_params["bondage"])

        if "bondage_hands" in agent_params and not agent_params.get("bondage_hands_prompt"):
            agent_params["bondage_hands_prompt"] = lookup_cat("bondage_hands", agent_params["bondage_hands"])

        if "bondage_legs" in agent_params and not agent_params.get("bondage_legs_prompt"):
            agent_params["bondage_legs_prompt"] = lookup_cat("bondage_legs", agent_params["bondage_legs"])

        if "bondage_accessories" in agent_params and not agent_params.get("bondage_accessories_prompt"):
            agent_params["bondage_accessories_prompt"] = lookup_cat("bondage_accessories", agent_params["bondage_accessories"])

        if "gag_type" in agent_params and not agent_params.get("gag_prompt"):
            agent_params["gag_prompt"] = lookup_cat("gag_types", agent_params["gag_type"])
        elif "gag" in agent_params and not agent_params.get("gag_prompt"):
            agent_params["gag_prompt"] = lookup_cat("gag_types", agent_params["gag"])

        if "scene" in agent_params and not agent_params.get("scene_prompt"):
            agent_params["scene_prompt"] = lookup_cat("scenes", agent_params["scene"])

        if "style" in agent_params and not agent_params.get("style_prompt"):
            agent_params["style_prompt"] = lookup_cat("styles", agent_params["style"])

        # Resolve character details
        char_id = params.get("character_id") or params.get("character")
        char_info = self.catalogue_manager.get_character_by_id(char_id) if char_id else None

        if char_info:
            agent_params["character_name"] = char_info["name"]
            agent_params["character_trigger"] = char_info.get("trigger", "")
            agent_params["character_description"] = char_info.get("description", "")

        return self.prompt_agent.generate_prompts(agent_params)

    def generate_single(
        self,
        params: Dict[str, Any],
        step_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes a single image generation:
        1. Calls Prompt Agent to synthesize prompt (or uses user-provided override prompt).
        2. Configures Hassaku XL Illustrious model and character LoRA adapter.
        3. Invokes the existing GenerationCoordinator (with optional High-Res Upscaler).
        """
        self.is_cancelled = False
        model_info = self.get_primary_model_info()

        # 1. Resolve prompt
        if params.get("positive_prompt"):
            prompts = {
                "positive_prompt": params["positive_prompt"],
                "negative_prompt": params.get("negative_prompt", self.prompt_agent._default_negative_prompt())
            }
        else:
            prompts = self.prepare_single_prompt(params)

        # 2. Resolve Character LoRA
        char_id = params.get("character_id") or params.get("character")
        char_info = self.catalogue_manager.get_character_by_id(char_id) if char_id else None
        
        loras_dict: Dict[str, float] = {}
        if char_info:
            lora_rel = char_info.get("lora_relative_path")
            lora_abs = char_info.get("lora_absolute_path")
            weight = float(params.get("character_weight", char_info.get("default_weight", 0.85)))
            if lora_abs and Path(lora_abs).exists():
                loras_dict[lora_abs] = weight
            elif lora_rel and (paths.LORAS_DIR / lora_rel).exists():
                loras_dict[str((paths.LORAS_DIR / lora_rel).resolve())] = weight
            else:
                print(f"[StoryStudio Pipeline] Character LoRA for '{char_id}' not found on disk. Proceeding without LoRA.")

        # 3. Seed handling
        raw_seed = params.get("seed", -1)
        if raw_seed is None or raw_seed == -1 or str(raw_seed).lower() == "random":
            actual_seed = random.randint(1, MAX_SEED)
        else:
            actual_seed = sanitize_seed(int(raw_seed))

        # 4. Upscaler resolution
        upscale_method = params.get("upscale_method")
        if upscale_method in ("None", "", None):
            upscale_method = None
        upscale_factor = float(params.get("upscale_factor", 2.0))

        # 5. Construct GenerationRequest
        req = GenerationRequest(
            model=model_info,
            prompt=prompts["positive_prompt"],
            negative_prompt=prompts["negative_prompt"],
            width=int(params.get("width", 1024)),
            height=int(params.get("height", 1536)),
            steps=int(params.get("steps", 30)),
            guidance_scale=float(params.get("cfg_scale", params.get("cfg", 5.5))),
            sampler=params.get("sampler", "Euler a"),
            scheduler=params.get("scheduler", "Normal"),
            seed=actual_seed,
            loras=loras_dict,
            upscale_method=upscale_method,
            upscale_factor=upscale_factor,
            step_callback=step_callback
        )

        # 5. Execute generation on existing engine
        res: GenerationResult = self.coordinator.generate(req)

        # Read image to data URL
        img_p = Path(res.image_path)
        with open(img_p, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64_img}"

        return {
            "image_url": data_url,
            "image_path": str(img_p.resolve()),
            "seed": actual_seed,
            "positive_prompt": prompts["positive_prompt"],
            "negative_prompt": prompts["negative_prompt"],
            "generation_time_ms": res.generation_time_ms,
            "vram_peak_gb": res.vram_peak_gb,
            "metadata": params
        }

    def generate_batch(
        self,
        params: Dict[str, Any],
        batch_count: int,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes batch generation sequentially.
        """
        self.is_cancelled = False
        results: List[Dict[str, Any]] = []

        for i in range(batch_count):
            if self.is_cancelled:
                break

            current_params = dict(params)
            # If seed was random, randomize each batch item
            if params.get("seed", -1) in (-1, None, "random"):
                current_params["seed"] = random.randint(1, MAX_SEED)

            res = self.generate_single(current_params)
            results.append(res)

            if progress_callback:
                progress_callback(i + 1, batch_count, res)

        return results

    def decompose_story_to_scenes(
        self,
        story_text: str,
        character_id: str,
        default_settings: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Decomposes a single raw story text into sequential narrative beats / scenes.
        For each scene:
        1. Breaks apart into logical sentences/paragraphs.
        2. Evaluates narrative continuity across scenes.
        3. Synthesizes positive and negative prompts formatted for Hassaku XL Illustrious.
        Returns a list of scene objects ready for user editing and generation.
        """
        if not story_text or not story_text.strip():
            return []

        raw_lines = self._split_story_text(story_text)
        if not raw_lines:
            return []

        char_info = self.catalogue_manager.get_character_by_id(character_id) if character_id else {}
        defaults = default_settings or {}

        scenes: List[Dict[str, Any]] = []
        tracker = ContinuityTracker()

        for idx, line_text in enumerate(raw_lines):
            continuity_state = tracker.get_state()

            # Call Story Agent
            directive = self.story_agent.interpret_story_line(
                current_line=line_text,
                line_index=idx,
                full_story=raw_lines,
                continuity_state=continuity_state,
                character_info=char_info,
                default_settings=defaults
            )

            # Update continuity
            tracker.update_from_directive(directive)

            # Call Prompt Agent
            prompts = self.prompt_agent.generate_prompts(directive)

            scenes.append({
                "index": idx,
                "text": line_text,
                "directive": directive,
                "positive_prompt": prompts["positive_prompt"],
                "negative_prompt": prompts["negative_prompt"],
                "status": "Pending",
                "image_url": None,
                "seed": -1
            })

        return scenes

    @staticmethod
    def _split_story_text(text: str) -> List[str]:
        """
        Smartly splits raw story text into sequential narrative beats.
        Supports paragraph breaks, dialogue, and numbered lines.
        """
        cleaned = text.strip()
        lines = []
        raw_paragraphs = [p.strip() for p in cleaned.split("\n") if p.strip()]
        for p in raw_paragraphs:
            p_clean = re.sub(r"^(?:scene\s*\d+[\s:\.\-]+|\d+[\.\)\-]\s*)", "", p, flags=re.IGNORECASE).strip()
            if not p_clean:
                continue
            if len(p_clean) > 120 and "." in p_clean:
                sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'“])", p_clean)
                for s in sentences:
                    s_clean = s.strip()
                    if s_clean:
                        lines.append(s_clean)
            else:
                lines.append(p_clean)

        return lines

    def generate_story_line(
        self,
        project_name: str,
        line_index: int,
        step_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Generates an individual story line:
        1. Loads project & context.
        2. Uses user-edited prompts or synthesizes them via Story Agent & Prompt Agent.
        3. Generates image via existing coordinator.
        4. Saves output to StoryStudio/projects/<ProjectName>/images/00X.png.
        5. Updates continuity state.
        """
        self.is_cancelled = False
        project = self.project_manager.get_project(project_name)
        if not project:
            raise ValueError(f"Project '{project_name}' not found.")

        lines = project.get("lines", [])
        if line_index < 0 or line_index >= len(lines):
            raise ValueError(f"Invalid line index {line_index} for project '{project_name}'.")

        target_line = lines[line_index]
        line_text = target_line.get("text", "")
        full_story_texts = [l.get("text", "") for l in lines]

        # 1. Update project status to Generating
        self.project_manager.update_line_status(project_name, line_index, "Generating")

        char_id = project.get("character")
        char_info = self.catalogue_manager.get_character_by_id(char_id) if char_id else {}
        defaults = project.get("default_settings", {})
        continuity_state = project.get("continuity") or self.continuity_tracker.get_state()

        try:
            # 2. Check if line has user-edited prompts or synthesize them
            custom_pos_prompt = target_line.get("positive_prompt")
            custom_neg_prompt = target_line.get("negative_prompt")

            if custom_pos_prompt and custom_pos_prompt.strip():
                pos_prompt = custom_pos_prompt
                neg_prompt = custom_neg_prompt or self.prompt_agent._default_negative_prompt()
                directive = target_line.get("directive") or {}
            else:
                directive = self.story_agent.interpret_story_line(
                    current_line=line_text,
                    line_index=line_index,
                    full_story=full_story_texts,
                    continuity_state=continuity_state,
                    character_info=char_info,
                    default_settings=defaults
                )
                prompts = self.prompt_agent.generate_prompts(directive)
                pos_prompt = prompts["positive_prompt"]
                neg_prompt = prompts["negative_prompt"]

            # 3. Generate Image via Coordinator
            gen_params = {
                "character_id": char_id,
                "positive_prompt": pos_prompt,
                "negative_prompt": neg_prompt,
                "width": defaults.get("width", 1024),
                "height": defaults.get("height", 1536),
                "steps": defaults.get("steps", 30),
                "cfg": defaults.get("cfg", 5.5),
                "sampler": defaults.get("sampler", "Euler a"),
                "scheduler": defaults.get("scheduler", "Normal"),
                "seed": defaults.get("seed", -1)
            }

            gen_result = self.generate_single(gen_params, step_callback=step_callback)

            # 4. Open image and save to project
            img = Image.open(gen_result["image_path"])

            # 5. Update Continuity Tracker
            if directive:
                self.continuity_tracker.deserialize(continuity_state)
                self.continuity_tracker.update_from_directive(directive)
                new_continuity = self.continuity_tracker.serialize()
            else:
                new_continuity = continuity_state

            # 6. Save to project
            prompt_meta = {
                "line_index": line_index,
                "line_text": line_text,
                "character": char_id,
                "directive": directive,
                "positive_prompt": pos_prompt,
                "negative_prompt": neg_prompt,
                "resolution": f"{gen_params['width']}x{gen_params['height']}",
                "steps": gen_params["steps"],
                "cfg": gen_params["cfg"],
                "sampler": gen_params["sampler"],
                "scheduler": gen_params["scheduler"],
                "seed": gen_result["seed"]
            }

            saved_img_path = self.project_manager.save_line_result(
                project_name=project_name,
                line_index=line_index,
                image=img,
                prompt_meta=prompt_meta,
                continuity_snapshot=new_continuity
            )

            return {
                "status": "Completed",
                "line_index": line_index,
                "image_url": gen_result["image_url"],
                "image_path": saved_img_path,
                "seed": gen_result["seed"],
                "directive": directive,
                "positive_prompt": pos_prompt,
                "negative_prompt": neg_prompt
            }

        except Exception as e:
            self.project_manager.update_line_status(project_name, line_index, "Failed", error=str(e))
            raise e

    def cancel(self):
        self.is_cancelled = True
