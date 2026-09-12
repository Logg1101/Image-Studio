import logging
from typing import Optional
from prompt_generator.types import (
    PromptGenerateRequest,
    PromptGenerateResponse,
    LLMSettings,
)
from prompt_generator.profiles import profile_registry
from prompt_generator.providers import provider_registry
from prompt_generator.parser import TagParser
from prompt_generator.settings import settings_manager

logger = logging.getLogger("PromptGenerator")

class PromptGenerator:
    """
    Coordinates LLM provider invocation, profile formatting,
    and deterministic tag parsing for SDXL generation.
    """

    def __init__(self):
        self.profile_registry = profile_registry
        self.provider_registry = provider_registry
        self.settings_manager = settings_manager

    async def generate_prompt(self, request: PromptGenerateRequest) -> PromptGenerateResponse:
        settings: LLMSettings = request.settings or self.settings_manager.get_settings()

        if not request.text or not request.text.strip():
            return PromptGenerateResponse(
                success=False,
                prompt="",
                error="Please enter a natural language description to generate tags.",
            )

        profile = self.profile_registry.get_profile(request.profile)
        provider = self.provider_registry.get_provider(settings.provider)

        try:
            # 1. Invoke LLM provider
            raw_output = await provider.generate(
                prompt=request.text.strip(),
                system_prompt=profile.system_prompt,
                settings=settings,
            )

            # 2. Deterministic tag parsing & deduplication
            parsed_tags = TagParser.parse_tags(raw_output)

            # 3. Model profile formatting (quality prefixes, style tags, suffixes)
            formatted_prompt = profile.format_prompt(parsed_tags, style=request.style)

            # 4. Merge with existing prompt if requested
            if request.existing_prompt and request.existing_prompt.strip():
                formatted_tags_list = TagParser.parse_tags(formatted_prompt)
                final_prompt = TagParser.merge_prompts(
                    existing_prompt=request.existing_prompt,
                    new_tags=formatted_tags_list,
                    mode=request.mode,
                )
            else:
                final_prompt = formatted_prompt

            return PromptGenerateResponse(
                success=True,
                prompt=final_prompt,
                negative_prompt=profile.negative_prompt,
                raw_tags=raw_output,
                parsed_tags=parsed_tags,
            )

        except Exception as e:
            logger.error(f"Prompt generation error: {e}", exc_info=True)
            return PromptGenerateResponse(
                success=False,
                prompt="",
                negative_prompt=profile.negative_prompt,
                error=str(e),
            )


prompt_generator = PromptGenerator()
