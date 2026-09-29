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
