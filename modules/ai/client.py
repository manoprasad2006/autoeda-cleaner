from __future__ import annotations

import time
from dataclasses import dataclass

from google import genai
from google.genai import types
from google.genai.errors import APIError


@dataclass(frozen=True)
class AIResponse:
    success: bool
    text: str
    error: str | None = None
    attempts: int = 1


class GeminiClient:
    def __init__(self, api_key: str, model: str, client: genai.Client | None = None):
        self.model = model
        self._client = (
            client
            if client is not None
            else genai.Client(
                api_key=api_key, http_options=types.HttpOptions(timeout=30000)
            )
        )

    def generate(self, prompt: str, max_retries: int = 3) -> AIResponse:
        last_error = ""

        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2, max_output_tokens=2500
                    ),
                )
                text = (response.text or "").strip()

                if not text:
                    last_error = "Model returned an empty response."
                    continue

                return AIResponse(success=True, text=text, attempts=attempt)

            except APIError as exc:
                last_error = "The AI service is temporarily unavailable."
                if getattr(exc, "code", None) == 429 and self.model != "gemini-3.1-flash-lite":
                    try:
                        fb_resp = self._client.models.generate_content(
                            model="gemini-3.1-flash-lite",
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                temperature=0.2, max_output_tokens=2500
                            ),
                        )
                        fb_text = (fb_resp.text or "").strip()
                        if fb_text:
                            return AIResponse(success=True, text=fb_text, attempts=attempt)
                    except Exception:
                        pass
                if getattr(exc, "code", None) in (400, 401, 403, 404):
                    return AIResponse(
                        success=False,
                        text="",
                        error="Check the configured AI key and model.",
                        attempts=attempt,
                    )
                if attempt < max_retries:
                    time.sleep(2 ** (attempt - 1))  # 1s, 2s, 4s...

            except Exception:  # noqa: BLE001
                # Anything not from the SDK itself (e.g. a bug in our own
                # code) shouldn't be silently retried -- fail immediately
                # with the real error rather than masking a real bug
                # behind three identical retry attempts.
                return AIResponse(
                    success=False,
                    text="",
                    error="The AI request could not be completed.",
                    attempts=attempt,
                )

        return AIResponse(
            success=False, text="", error=last_error, attempts=max_retries
        )


class GroqRotationClient:
    """Thread-safe Groq client rotating across multiple API keys with automatic failover."""

    def __init__(
        self,
        api_keys: list[str] | tuple[str, ...],
        model: str = "openai/gpt-oss-120b",
        fallback_models: list[str] | None = None,
    ):
        import threading
        self.api_keys = [k.strip() for k in api_keys if k and k.strip()]
        self.model = model
        self.fallback_models = fallback_models or ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
        self._index = 0
        self._lock = threading.Lock()
        self.last_key_number = 1

    def _next_key_info(self) -> tuple[str, int]:
        with self._lock:
            if not self.api_keys:
                return "", 0
            idx = self._index % len(self.api_keys)
            key = self.api_keys[idx]
            self._index = (self._index + 1) % len(self.api_keys)
            self.last_key_number = idx + 1
            return key, idx + 1

    def generate(self, prompt: str, max_retries: int = 3) -> AIResponse:
        from groq import Groq

        if not self.api_keys:
            return AIResponse(success=False, text="", error="No Groq API keys configured.")

        models_to_try = [self.model] + [m for m in self.fallback_models if m != self.model]
        attempts = 0
        last_error = ""

        # Cycle through keys up to max_retries times
        for _ in range(max(1, min(max_retries, len(self.api_keys)))):
            api_key, key_num = self._next_key_info()
            if not api_key:
                continue

            try:
                client = Groq(api_key=api_key)
                for m in models_to_try:
                    attempts += 1
                    try:
                        resp = client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model=m,
                            temperature=0.2,
                            max_tokens=2500,
                        )
                        text = (resp.choices[0].message.content or "").strip()
                        if text:
                            return AIResponse(success=True, text=text, attempts=attempts)
                    except Exception as exc:
                        last_error = str(exc)
                        # If model not found or rate limit, try fallback model or next key
                        continue
            except Exception as exc:
                last_error = str(exc)
                continue

        return AIResponse(
            success=False,
            text="",
            error=f"Groq API error across rotated keys: {last_error}",
            attempts=attempts,
        )


def get_ai_client() -> GroqRotationClient | GeminiClient | None:
    """Return configured AI client prioritizing Groq with key rotation, then Gemini."""
    from utils.config import load_settings

    settings = load_settings()
    if settings.has_groq:
        return GroqRotationClient(api_keys=settings.groq_keys, model=settings.groq_model)
    elif settings.gemini_api_key:
        return GeminiClient(api_key=settings.gemini_api_key, model=settings.gemini_model)
    return None
