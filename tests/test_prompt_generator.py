import unittest
import asyncio
from unittest.mock import patch, MagicMock
import httpx

from prompt_generator.types import LLMSettings, PromptGenerateRequest
from prompt_generator.parser import TagParser
from prompt_generator.profiles import (
    profile_registry,
    SDXL_BASE_PROFILE,
    ILLUSTRIOUS_XL_PROFILE,
    PONY_PROFILE,
    ANIMAGINE_XL_PROFILE,
)
from prompt_generator.providers import OpenAICompatibleProvider
from prompt_generator.generator import PromptGenerator


class TestTagParser(unittest.TestCase):
    def test_basic_parsing_and_deduplication(self):
        raw = "1girl, black hair,  black hair, , rainy street, tokyo"
        parsed = TagParser.parse_tags(raw)
        expected = ["1girl", "black hair", "rainy street", "tokyo"]
        self.assertEqual(parsed, expected)

    def test_case_insensitive_deduplication(self):
        raw = "1girl, Black Hair, black hair, RAINY STREET, rainy street"
        parsed = TagParser.parse_tags(raw)
        # Keeps first occurrence casing
        self.assertEqual(parsed, ["1girl", "Black Hair", "RAINY STREET"])

    def test_weighting_syntax_preserved(self):
        raw = "(1girl:1.2), [black hair], (rainy street:0.9), (masterpiece)"
        parsed = TagParser.parse_tags(raw)
        self.assertIn("(1girl:1.2)", parsed)
        self.assertIn("[black hair]", parsed)
        self.assertIn("(rainy street:0.9)", parsed)
        self.assertIn("(masterpiece)", parsed)

    def test_lora_syntax_preserved(self):
        raw = "1girl, <lora:cyberpunk_style:0.8>, neon lighting, <lora:tokyo_rain:1.0>"
        parsed = TagParser.parse_tags(raw)
        self.assertIn("<lora:cyberpunk_style:0.8>", parsed)
        self.assertIn("<lora:tokyo_rain:1.0>", parsed)

    def test_markdown_codeblock_stripping(self):
        raw = "```text\n1girl, black hair, rainy street\n```"
        parsed = TagParser.parse_tags(raw)
        self.assertEqual(parsed, ["1girl", "black hair", "rainy street"])

    def test_merge_prompts_replace(self):
        existing = "masterpiece, cinematic photograph"
        new_tags = ["1girl", "black hair", "rainy street"]
        merged = TagParser.merge_prompts(existing, new_tags, mode="replace")
        self.assertEqual(merged, "1girl, black hair, rainy street")

    def test_merge_prompts_append_preserves_lora_and_no_dupes(self):
        existing = "masterpiece, <lora:anime_v1:0.8>, cinematic photograph"
        new_tags = ["1girl", "black hair", "masterpiece", "rainy street"]
        merged = TagParser.merge_prompts(existing, new_tags, mode="append")
        expected = "masterpiece, <lora:anime_v1:0.8>, cinematic photograph, 1girl, black hair, rainy street"
        self.assertEqual(merged, expected)


class TestModelProfiles(unittest.TestCase):
    def test_profile_retrieval(self):
        self.assertEqual(profile_registry.get_profile("joycaption").id, "joycaption")
        self.assertEqual(profile_registry.get_profile("sdxl_base").id, "sdxl_base")
        self.assertEqual(profile_registry.get_profile("illustrious_xl").id, "illustrious_xl")
        self.assertEqual(profile_registry.get_profile("pony").id, "pony")
        self.assertEqual(profile_registry.get_profile("animagine_xl").id, "animagine_xl")
        # Fallback to default
        self.assertEqual(profile_registry.get_profile("unknown_model").id, "joycaption")

    def test_joycaption_formatting(self):
        clauses = [
            "A detailed medium shot of a silver-haired young woman in a tailored emerald velvet blazer",
            "standing on a rain-slicked cobblestone street in neon-lit Kyoto",
        ]
        formatted = profile_registry.get_profile("joycaption").format_prompt(clauses, style="General")
        self.assertIn("silver-haired young woman", formatted)
        self.assertIn("rain-slicked cobblestone street", formatted)
        self.assertIn("high resolution, natural skin texture, depth of field", formatted)

    def test_sdxl_base_formatting(self):
        tags = ["1girl", "black hair", "rainy street"]
        formatted = SDXL_BASE_PROFILE.format_prompt(tags, style="General")
        self.assertIn("masterpiece, highly detailed, best quality", formatted)
        self.assertIn("1girl, black hair, rainy street", formatted)
        self.assertIn("cinematic lighting, sharp focus", formatted)

    def test_pony_formatting(self):
        tags = ["1girl", "solo", "black_hair"]
        formatted = PONY_PROFILE.format_prompt(tags, style="General")
        self.assertTrue(formatted.startswith("score_9, score_8_up, score_7_up, source_anime, rating_safe"))
        self.assertIn("1girl, solo, black_hair", formatted)


class TestOpenAICompatibleProvider(unittest.TestCase):
    def setUp(self):
        self.provider = OpenAICompatibleProvider()
        self.settings = LLMSettings(
            base_url="http://127.0.0.1:11434/v1",
            model="qwen2.5:7b",
            temperature=0.2,
            max_tokens=256,
            timeout=5,
        )

    def test_valid_response(self):
        async def run():
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "1girl, black hair, rainy street, neon lights"}}]
            }

            with patch("httpx.AsyncClient.post", return_value=mock_resp):
                res = await self.provider.generate(
                    prompt="woman on a rainy street",
                    system_prompt="convert to tags",
                    settings=self.settings,
                )
                self.assertEqual(res, "1girl, black hair, rainy street, neon lights")

        asyncio.run(run())

    def test_connection_failure(self):
        async def run():
            with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection refused")):
                with self.assertRaises(ConnectionError) as ctx:
                    await self.provider.generate(
                        prompt="woman on a rainy street",
                        system_prompt="convert to tags",
                        settings=self.settings,
                    )
                self.assertIn("LLM server unavailable", str(ctx.exception))

        asyncio.run(run())

    def test_timeout_failure(self):
        async def run():
            with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Timed out")):
                with self.assertRaises(TimeoutError) as ctx:
                    await self.provider.generate(
                        prompt="woman on a rainy street",
                        system_prompt="convert to tags",
                        settings=self.settings,
                    )
                self.assertIn("timed out", str(ctx.exception))

        asyncio.run(run())

    def test_malformed_response(self):
        async def run():
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"error": "unexpected format"}

            with patch("httpx.AsyncClient.post", return_value=mock_resp):
                with self.assertRaises(ValueError):
                    await self.provider.generate(
                        prompt="woman on a rainy street",
                        system_prompt="convert to tags",
                        settings=self.settings,
                    )

        asyncio.run(run())


class TestEndToEndGenerator(unittest.TestCase):
    def test_full_generation_pipeline(self):
        async def run():
            generator = PromptGenerator()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{
                    "message": {
                        "content": "1girl, black hair, rainy street, neon lights, reflections"
                    }
                }]
            }

            req = PromptGenerateRequest(
                text="A young woman with black hair standing on a rainy street with neon lights",
                profile="sdxl_base",
                style="Cinematic Photograph",
                existing_prompt="masterpiece, <lora:tokyo_night:0.8>",
                mode="append",
                settings=LLMSettings(base_url="http://127.0.0.1:11434/v1", model="test-model"),
            )

            with patch("httpx.AsyncClient.post", return_value=mock_resp):
                resp = await generator.generate_prompt(req)
                self.assertTrue(resp.success)
                self.assertIn("<lora:tokyo_night:0.8>", resp.prompt)
                self.assertIn("1girl", resp.prompt)
                self.assertIn("black hair", resp.prompt)
                self.assertIn("rainy street", resp.prompt)
                self.assertIn("photorealistic", resp.prompt)
                self.assertTrue(len(resp.parsed_tags) >= 5)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
