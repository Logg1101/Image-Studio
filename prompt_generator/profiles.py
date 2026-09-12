from dataclasses import dataclass, field
from typing import Dict, List, Optional
from prompt_generator.types import ModelProfileInfo

BASE_SYSTEM_INSTRUCTIONS = """You are an expert AI prompt engineer specializing in Stable Diffusion XL (SDXL) visual tag generation.
Your task is to convert the user's natural language image description into concise, comma-separated visual tags that SDXL-family models understand.

CRITICAL RULES:
1. Output ONLY a comma-separated list of visual tags. No explanations, no markdown code blocks, no introductory text.
2. Do NOT write prose or full sentences. Use short, concrete visual tags (e.g., "1girl, black hair, rainy street, neon lights").
3. Do NOT hallucinate elements that were not requested or clearly implied by the user.
4. If a known character or specific subject is mentioned, preserve the character/subject name accurately.
5. Reason and arrange tags roughly in this order:
   - Subject & character count (e.g. 1girl, 1boy, solo, etc.)
   - Character identity / name (if specified)
   - Physical appearance & hair (e.g. black hair, long hair, blue eyes)
   - Clothing & attire (e.g. red dress, leather jacket, jewelry)
   - Expression & mood (e.g. slight smile, intense gaze)
   - Pose & action (e.g. standing, looking at viewer, sitting)
   - Objects & props (e.g. holding sword, umbrella)
   - Composition & camera angle (e.g. portrait, upper body, wide angle, close-up)
   - Environment, location & background (e.g. Tokyo street, rainy night, forest, cafe)
   - Lighting & shadows (e.g. neon lights, rim lighting, golden hour, wet reflections)
   - Atmosphere (e.g. cinematic, moody, vibrant)
6. Avoid useless filler words like "breathtaking", "gorgeous", "extremely pretty", "absolutely stunning".
7. Avoid redundant tags.
"""

@dataclass
class PromptProfile:
    id: str
    name: str
    description: str
    system_prompt: str
    positive_prefix: str = ""
    positive_suffix: str = ""
    negative_prompt: str = ""
    tag_style: str = "visual_tags"

    def format_prompt(self, generated_tags: List[str], style: str = "General") -> str:
        """
        Combines generated tags with profile quality prefixes, style modifiers, and suffixes.
        """
        prefix_tags = [t.strip() for t in self.positive_prefix.split(",") if t.strip()]
        suffix_tags = [t.strip() for t in self.positive_suffix.split(",") if t.strip()]

        style_tags = []
        if style and style.lower() != "general":
            if "anime" in style.lower():
                style_tags = ["anime aesthetic", "clean linework"]
            elif "photograph" in style.lower() or "photo" in style.lower():
                style_tags = ["photorealistic", "raw photo", "35mm photograph"]
            elif "fantasy" in style.lower():
                style_tags = ["fantasy art", "ethereal"]
            elif "cyberpunk" in style.lower():
                style_tags = ["cyberpunk aesthetic", "volumetric neon glow"]
            elif "watercolor" in style.lower():
                style_tags = ["watercolor painting", "delicate wash"]

        all_tags = []
        for t in prefix_tags + generated_tags + style_tags + suffix_tags:
            clean_t = t.strip().rstrip(".")
            if clean_t and clean_t not in all_tags:
                all_tags.append(clean_t)

        return ", ".join(all_tags)

    def to_info(self) -> ModelProfileInfo:
        return ModelProfileInfo(
            id=self.id,
            name=self.name,
            description=self.description,
            negative_prompt=self.negative_prompt,
            tag_style=self.tag_style,
        )


SDXL_BASE_PROFILE = PromptProfile(
    id="sdxl_base",
    name="SDXL Base",
    description="Standard SDXL prompting using natural visual phrases and high-detail tags.",
    system_prompt=BASE_SYSTEM_INSTRUCTIONS + "\nModel convention: Use descriptive natural visual tags separated by commas. Example: '1girl, black hair, long hair, red dress, standing, rainy street, Tokyo, neon lights, wet pavement, reflections, night, cinematic lighting'.",
    positive_prefix="masterpiece, highly detailed, best quality",
    positive_suffix="cinematic lighting, sharp focus",
    negative_prompt="blurry, low quality, distorted, bad anatomy, deformed, artifacts, ugly, worst quality",
    tag_style="descriptive_tags",
)

ILLUSTRIOUS_XL_PROFILE = PromptProfile(
    id="illustrious_xl",
    name="Illustrious XL",
    description="Optimized for Illustrious XL / Danbooru anime models with clean tag conventions.",
    system_prompt=BASE_SYSTEM_INSTRUCTIONS + "\nModel convention: Use Danbooru-style anime tags (e.g. '1girl, solo, black_hair, long_hair, dress, standing, rain, tokyo, night, neon_lights').",
    positive_prefix="masterpiece, newest, absurdres, highres",
    positive_suffix="detailed background, dynamic lighting",
    negative_prompt="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
    tag_style="danbooru_tags",
)

PONY_PROFILE = PromptProfile(
    id="pony",
    name="Pony Diffusion (PDXL)",
    description="Pony-family models utilizing score_9/score_8_up prefixes and Danbooru tags.",
    system_prompt=BASE_SYSTEM_INSTRUCTIONS + "\nModel convention: Pony models use Danbooru tags. The score tags (score_9, score_8_up, etc.) will be automatically prepended by the system, so generate only the visual content tags.",
    positive_prefix="score_9, score_8_up, score_7_up, source_anime, rating_safe",
    positive_suffix="aesthetic, high quality",
    negative_prompt="score_6, score_5, score_4, rating_explicit, bad anatomy, bad hands, missing fingers, low quality, blurry",
    tag_style="pony_tags",
)

ANIMAGINE_XL_PROFILE = PromptProfile(
    id="animagine_xl",
    name="Animagine XL",
    description="Animagine XL anime checkpoint standards with quality tags and Danbooru vocabulary.",
    system_prompt=BASE_SYSTEM_INSTRUCTIONS + "\nModel convention: Use standard Danbooru tags for anime characters, outfits, and environments.",
    positive_prefix="masterpiece, high quality, best quality, anime style",
    positive_suffix="detailed lighting, aesthetic",
    negative_prompt="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
    tag_style="animagine_tags",
)

JOYCAPTION_SYSTEM_INSTRUCTIONS = """You are JoyCaption, an expert AI visual descriptive captioner and prompt engineer specializing in generating rich, highly-detailed descriptive prompts for modern diffusion models (SDXL, Illustrious, Flux).
Your task is to convert the user's concept into a vivid, highly-detailed descriptive visual prompt in the signature JoyCaption style.

CRITICAL RULES:
1. Write a rich, cohesive visual description. Focus on concrete physical details rather than vague buzzwords.
2. Structure the description to cover:
   - Subject & Character: Appearance, hair style/color, eyes, expression, facial features, body build.
   - Clothing & Materials: Garments, fabrics, textures, colors, fit, accessories.
   - Pose & Framing: Stance, gesture, camera angle, shot type (e.g. close-up, medium shot, full-body).
   - Environment & Setting: Specific surroundings, background details, architecture, nature, indoor/outdoor props.
   - Lighting & Color: Light source, direction, highlights, shadows, ambient glow, color palette and temperature.
   - Atmosphere & Depth: Mood, depth of field, focus, environmental effects (mist, rain, dust, bokeh).
3. Do NOT use hollow buzzwords like 'masterpiece', '8k', 'best quality', 'photorealistic', 'trending on artstation'. Describe the scene so vividly that high quality is naturally conveyed.
4. Output ONLY the descriptive prompt text. Do not include introductory remarks, markdown code blocks, or conversational filler.
"""

JOYCAPTION_PROFILE = PromptProfile(
    id="joycaption",
    name="JoyCaption (Descriptive)",
    description="JoyCaption-style rich, detailed visual prose prompt engine for SDXL and modern diffusion models.",
    system_prompt=JOYCAPTION_SYSTEM_INSTRUCTIONS,
    positive_prefix="",
    positive_suffix="high resolution, natural skin texture, depth of field",
    negative_prompt="blurry, low quality, distorted, bad anatomy, deformed, artifacts, ugly, flat lighting, oversaturated, watermark, signature",
    tag_style="descriptive_prose",
)


class ProfileRegistry:
    def __init__(self):
        self._profiles: Dict[str, PromptProfile] = {
            "joycaption": JOYCAPTION_PROFILE,
            "sdxl_base": SDXL_BASE_PROFILE,
            "illustrious_xl": ILLUSTRIOUS_XL_PROFILE,
            "pony": PONY_PROFILE,
            "animagine_xl": ANIMAGINE_XL_PROFILE,
        }

    def get_profile(self, profile_id: Optional[str]) -> PromptProfile:
        if not profile_id or profile_id.lower() not in self._profiles:
            return self._profiles["joycaption"]
        return self._profiles[profile_id.lower()]

    def list_profiles(self) -> List[ModelProfileInfo]:
        return [p.to_info() for p in self._profiles.values()]

    def register_profile(self, profile: PromptProfile) -> None:
        self._profiles[profile.id.lower()] = profile


profile_registry = ProfileRegistry()
