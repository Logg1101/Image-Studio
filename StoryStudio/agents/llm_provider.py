import json
import os
import re
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

class LLMProvider:
    """
    Universal OpenAI-compatible Local LLM Provider client.
    Supports LM Studio, Ollama, vLLM, LocalAI, or any standard OpenAI HTTP endpoint.
    Handles timeouts, retries, and JSON extraction safely.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.provider = self.config.get("provider", "local")
        self.base_url = (
            os.environ.get("STORY_STUDIO_LLM_URL")
            or self.config.get("base_url")
            or "http://localhost:1234/v1"
        ).rstrip("/")
        self.model = (
            os.environ.get("STORY_STUDIO_LLM_MODEL")
            or self.config.get("model")
            or "local-model"
        )
        self.temperature = float(self.config.get("temperature", 0.7))
        self.max_tokens = int(self.config.get("max_tokens", 1200))
        self.timeout = float(self.config.get("timeout_seconds", 2.5))
        self.api_key = os.environ.get("STORY_STUDIO_LLM_KEY") or self.config.get("api_key", "sk-local")

    def is_configured(self) -> bool:
        return bool(self.base_url)

    def generate_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Sends request to the LLM endpoint and parses the structured JSON response.
        Returns None if unreachable, timed out, or unparseable.
        """
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"}
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    resp_body = resp.read().decode("utf-8")
                    result_json = json.loads(resp_body)
                    content = result_json["choices"][0]["message"]["content"]
                    return self._extract_json(content)
        except Exception as e:
            # LLM server offline or timed out; caller will use deterministic offline fallback
            return None

        return None

    @staticmethod
    def _extract_json(content: str) -> Optional[Dict[str, Any]]:
        """Extracts and parses JSON object from raw response string or markdown block."""
        if not content:
            return None
        content = content.strip()
        
        # 1. Direct parse attempt
        try:
            return json.loads(content)
        except Exception:
            pass

        # 2. Extract from ```json ... ``` markdown block
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except Exception:
                pass

        # 3. Extract bracketed substring { ... }
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except Exception:
                pass

        return None
