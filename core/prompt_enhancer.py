import random

class AIPromptEnhancer:
    SDXL_QUALITY_TAGS = [
        "masterpiece, best quality, ultra-detailed",
        "cinematic lighting, 8k resolution, sharp focus, masterpiece",
        "intricate details, highly detailed textures, professional illustration",
        "dramatic lighting, rich color palette, polished composition"
    ]
    SDXL_LIGHTING = [
        "soft volumetric lighting, warm ambient glow",
        "dramatic rim light, deep shadows, cinematic atmosphere",
        "natural sunlight, golden hour, soft diffusion"
    ]
    FLUX_PROSE_PREFIXES = [
        "A highly detailed and atmospheric photograph of",
        "A stunning, intricate masterpiece depicting",
        "A cinematic portrait capturing",
        "An exquisitely rendered, high-detail illustration of"
    ]
    FLUX_PROSE_DETAILS = [
        "rendered with soft volumetric lighting, subtle shadow gradients, and crisp focal depth",
        "featuring intricate fabric textures, natural proportions, and cinematic atmospheric depth",
        "illuminated by gentle ambient glow, with delicate surface details and rich color harmony"
    ]
    @classmethod
    def enhance(cls, prompt: str, architecture: str = "sdxl") -> str:
        prompt_clean = prompt.strip() if prompt else ""
        if not prompt_clean:
            prompt_clean = "a beautiful anime character in a serene setting"
        arch = architecture.lower() if architecture else "sdxl"
        if "flux" in arch:
            prefix = random.choice(cls.FLUX_PROSE_PREFIXES)
            detail = random.choice(cls.FLUX_PROSE_DETAILS)
            if prompt_clean.lower().startswith(("a ", "an ", "the ")):
                return f"{prompt_clean}, {detail}."
            else:
                return f"{prefix} {prompt_clean}, {detail}."
        else:
            quality = random.choice(cls.SDXL_QUALITY_TAGS)
            lighting = random.choice(cls.SDXL_LIGHTING)
            return f"{prompt_clean}, {quality}, {lighting}"
