"""Runtime configuration, read from environment variables (and backend/.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_STATIC_DIR = BACKEND_DIR.parent / "frontend" / "dist"
DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env_str(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    value = _env_str(name)
    if value is None:
        return default
    try:
        return max(minimum, int(value))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = _env_str(name)
    if value is None:
        return default
    return tuple(item.strip().rstrip("/") for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"
    serpapi_key: str | None = None
    google_maps_api_key: str | None = None
    demo: bool = False

    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    trust_proxy: bool = False
    static_dir: str | None = str(DEFAULT_STATIC_DIR)

    rate_limit_per_minute: int = 20
    max_message_chars: int = 2000
    max_history_messages: int = 12
    max_history_chars: int = 8000
    max_places: int = 25
    nominatim_user_agent: str = "MapChat/2.0 (self-hosted)"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openai_api_key=_env_str("OPENAI_API_KEY"),
            openai_model=_env_str("OPENAI_MODEL", "gpt-4o") or "gpt-4o",
            # SERPAPI_KEY is the documented name; SERP_API_KEY is what agno reads natively.
            serpapi_key=_env_str("SERPAPI_KEY") or _env_str("SERP_API_KEY"),
            google_maps_api_key=_env_str("GOOGLE_MAPS_API_KEY"),
            demo=_env_bool("MAPCHAT_DEMO"),
            host=_env_str("MAPCHAT_HOST", "127.0.0.1") or "127.0.0.1",
            port=_env_int("MAPCHAT_PORT", 5000, minimum=1),
            debug=_env_bool("MAPCHAT_DEBUG", _env_bool("FLASK_DEBUG")),
            cors_origins=_env_list("MAPCHAT_CORS_ORIGINS", DEFAULT_CORS_ORIGINS),
            trust_proxy=_env_bool("MAPCHAT_TRUST_PROXY"),
            static_dir=_env_str("MAPCHAT_STATIC_DIR", str(DEFAULT_STATIC_DIR)),
            rate_limit_per_minute=_env_int("MAPCHAT_RATE_LIMIT", 20),
            max_message_chars=_env_int("MAPCHAT_MAX_MESSAGE_CHARS", 2000, minimum=1),
            max_history_messages=_env_int("MAPCHAT_MAX_HISTORY", 12),
            max_places=_env_int("MAPCHAT_MAX_PLACES", 25, minimum=1),
            nominatim_user_agent=_env_str("NOMINATIM_USER_AGENT", "MapChat/2.0 (self-hosted)")
            or "MapChat/2.0 (self-hosted)",
        )

    @property
    def llm_configured(self) -> bool:
        return self.demo or bool(self.openai_api_key)

    @property
    def search_enabled(self) -> bool:
        return not self.demo and bool(self.serpapi_key)

    @property
    def geocoder_name(self) -> str:
        if self.demo:
            return "demo"
        return "google" if self.google_maps_api_key else "nominatim"
