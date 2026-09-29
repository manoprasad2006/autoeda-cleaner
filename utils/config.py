from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env into the process environment. Safe to call even if .env
# doesn't exist (e.g. in production, where real env vars are injected
# by the hosting platform instead of a file).
load_dotenv()

# Absolute path to the project root (this file lives in utils/, so root
# is one level up). Using an absolute path means file paths work the
# same whether the app is launched from the project root or elsewhere.
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """
    Immutable settings object. `frozen=True` means once created, its
    fields can't be reassigned — that prevents a bug where some module
    accidentally mutates shared config at runtime.
    """

    # --- AI ---
    gemini_api_key: str
    gemini_model: str

    # --- App behavior ---
    app_env: str
    log_level: str
    max_upload_mb: int

    # --- Paths ---
    base_dir: Path
    uploads_dir: Path
    outputs_dir: Path
    reports_dir: Path
    logs_dir: Path

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


def _get_int(name: str, default: int) -> int:
    """Read an env var as int, falling back to a default if unset or invalid."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        # Fail soft on bad config rather than crashing app startup —
        # we'd rather run with a sane default and log a warning later.
        return default


def load_settings() -> Settings:
    """
    Build a Settings instance from the current environment.

    Called once at app startup (see app.py). Deliberately does NOT
    cache/memoize here — Streamlit reruns the script on every
    interaction, so we rely on st.cache_resource at the call site
    instead of hiding caching logic inside config.
    """
    settings = Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        app_env=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        max_upload_mb=max(1, min(50, _get_int("MAX_UPLOAD_MB", 50))),
        base_dir=BASE_DIR,
        uploads_dir=BASE_DIR / "uploads",
        outputs_dir=BASE_DIR / "outputs",
        reports_dir=BASE_DIR / "reports",
        logs_dir=BASE_DIR / "logs",
    )

    # Ensure runtime directories exist. Doing this here means every
    # other module can assume these paths are safe to write to.
    for directory in (
        settings.uploads_dir,
        settings.outputs_dir,
        settings.reports_dir,
        settings.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return settings
