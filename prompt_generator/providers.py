import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import httpx

from prompt_generator.types import LLMSettings

logger = logging.getLogger("LLMProvider")

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    Supports asynchronous generation, connectivity testing, and optional lifecycle hooks.
    """

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str, settings: LLMSettings) -> str:
        """Generates visual tags from input text using the configured LLM."""
        pass

    @abstractmethod
    async def test_connection(self, settings: LLMSettings) -> Dict[str, Any]:
        """Verifies connection to backend server and retrieves available models."""
        pass

    async def load(self) -> None:
        """Optional lifecycle hook if backend requires explicit model loading."""
        pass

    async def unload(self) -> None:
        """Optional lifecycle hook if backend requires explicit model unloading / VRAM release."""
        pass


class OpenAICompatibleProvider(LLMProvider):
    """
    Provider for any OpenAI-compatible HTTP server:
    Ollama, llama.cpp server, LM Studio, vLLM, VRAM-Zero, Text-Gen-WebUI, LocalAI, etc.
    """

    def _get_headers(self, settings: LLMSettings) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if settings.api_key and settings.api_key.strip():
            headers["Authorization"] = f"Bearer {settings.api_key.strip()}"
        return headers

    def _normalize_base_url(self, base_url: str) -> str:
        url = base_url.strip().rstrip("/")
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"http://{url}"
        # Standard OpenAI compatible endpoints are mounted under /v1
        if not url.endswith("/v1") and not "/v1/" in url:
            url = f"{url}/v1"
        return url

    async def _discover_active_server(self, primary_url: str, headers: Dict[str, str]) -> Optional[str]:
        """Probes primary and known local server ports (Ollama:11434, LMStudio:1234, VRAM-Zero:8400)."""
        candidates = [primary_url]
        for fallback in ["http://127.0.0.1:11434/v1", "http://127.0.0.1:1234/v1", "http://127.0.0.1:8400/v1"]:
            if fallback not in candidates:
                candidates.append(fallback)

        async with httpx.AsyncClient(timeout=httpx.Timeout(1.5, connect=1.0)) as client:
            for candidate in candidates:
                try:
                    res = await client.get(f"{candidate}/models", headers=headers)
                    if res.status_code == 200:
                        return candidate
                except Exception:
                    continue
        return None

    async def _get_available_models(self, base_url: str, headers: Dict[str, str]) -> List[str]:
        """Queries /models on the target server."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(3.0, connect=2.0)) as client:
                res = await client.get(f"{base_url}/models", headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    models = []
                    for item in data.get("data", []):
                        if isinstance(item, dict) and "id" in item:
                            models.append(item["id"])
                        elif isinstance(item, str):
                            models.append(item)
                    return models
        except Exception:
            pass
        return []

    async def generate(self, prompt: str, system_prompt: str, settings: LLMSettings) -> str:
        base_url = self._normalize_base_url(settings.base_url)
        headers = self._get_headers(settings)

        endpoint = f"{base_url}/chat/completions"
        payload = {
            "model": settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(settings.temperature),
            "max_tokens": int(settings.max_tokens),
            "stream": False,
        }

        timeout = httpx.Timeout(settings.timeout, connect=10.0)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(endpoint, headers=headers, json=payload)

        except httpx.ConnectError as e:
            # Primary server failed to connect. Try fallback discovery of local active servers (VRAM-Zero:8400, Ollama:11434, LM Studio:1234)
            discovered = await self._discover_active_server(base_url, headers)
            if discovered and discovered != base_url:
                logger.info(f"[PromptGenerator] Switched to active server: {discovered}")
                settings.base_url = discovered
                endpoint = f"{discovered}/chat/completions"
                # Discover valid model if needed
                models = await self._get_available_models(discovered, headers)
                if models and (not payload["model"] or payload["model"] not in models):
                    payload["model"] = next(
                        (m for m in models if "joycaption" in m.lower() or "caption" in m.lower() or "qwen" in m.lower() or "gemma" in m.lower()),
                        models[0],
                    )
                try:
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        response = await client.post(endpoint, headers=headers, json=payload)
                except Exception as ex:
                    raise ConnectionError(
                        f"LLM server unavailable at {base_url} (and fallback {discovered} failed: {ex})."
                    ) from ex
            else:
                raise ConnectionError(
                    f"LLM server unavailable at {base_url}. "
                    f"Please ensure your JoyCaption / LLM server (e.g. Ollama on port 11434 or LM Studio on port 1234) is running."
                ) from e

        except httpx.TimeoutException as e:
            raise TimeoutError(
                f"LLM server at {base_url} timed out after {settings.timeout} seconds."
            ) from e

        except (ConnectionError, TimeoutError, RuntimeError, ValueError):
            raise

        except Exception as e:
            raise RuntimeError(f"Failed to communicate with LLM provider: {str(e)}") from e

        if response.status_code != 200:
            detail = response.text
            try:
                err_json = response.json()
                detail = err_json.get("error", {}).get("message", detail)
            except Exception:
                pass
            raise RuntimeError(
                f"LLM server returned HTTP {response.status_code}: {detail}"
            )

        data = response.json()
        choices = data.get("choices", [])
        if not choices or not isinstance(choices, list):
            raise ValueError("Malformed response from LLM server: missing 'choices' list.")

        first_choice = choices[0]
        message = first_choice.get("message", {})
        content = message.get("content", "")

        if not content and "text" in first_choice:
            content = first_choice["text"]

        if not content:
            raise ValueError("LLM server returned an empty content string.")

        return content.strip()

    async def test_connection(self, settings: LLMSettings) -> Dict[str, Any]:
        base_url = self._normalize_base_url(settings.base_url)
        headers = self._get_headers(settings)
        timeout = httpx.Timeout(8.0, connect=4.0)

        # 1. Try GET /models endpoint
        models_endpoint = f"{base_url}/models"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                res = await client.get(models_endpoint, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    model_list: List[str] = []
                    for item in data.get("data", []):
                        if isinstance(item, dict) and "id" in item:
                            model_list.append(item["id"])
                        elif isinstance(item, str):
                            model_list.append(item)
                    return {
                        "success": True,
                        "base_url": base_url,
                        "models": sorted(model_list),
                        "message": f"Connected to {base_url}. Found {len(model_list)} models.",
                    }
        except Exception:
            pass

        # 2. Check if another local server is active
        discovered = await self._discover_active_server(base_url, headers)
        if discovered and discovered != base_url:
            models = await self._get_available_models(discovered, headers)
            return {
                "success": True,
                "base_url": discovered,
                "models": sorted(models),
                "message": f"Found active LLM server at {discovered}! (Update base URL to use this).",
            }

        return {
            "success": False,
            "base_url": base_url,
            "error": f"Connection failed at {base_url}. Make sure your JoyCaption / LLM server (e.g. Ollama on port 11434 or LM Studio on port 1234) is started.",
        }


class ProviderRegistry:
    def __init__(self):
        self._providers: Dict[str, LLMProvider] = {
            "openai_compatible": OpenAICompatibleProvider(),
        }

    def get_provider(self, provider_id: Optional[str]) -> LLMProvider:
        pid = (provider_id or "openai_compatible").lower()
        if pid not in self._providers:
            # Default to openai_compatible
            return self._providers["openai_compatible"]
        return self._providers[pid]

    def register_provider(self, provider_id: str, provider: LLMProvider) -> None:
        self._providers[provider_id.lower()] = provider


provider_registry = ProviderRegistry()
