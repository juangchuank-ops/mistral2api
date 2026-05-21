"""Configuration management for mistral2api."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class AppConfig:
    """Application configuration."""
    api_keys: List[str] = field(default_factory=lambda: ["sk-mistral2api"])
    host: str = "0.0.0.0"
    port: int = 8000
    base_url: str = "https://chat.mistral.ai"
    default_model: str = "mistral-large-latest"
    request_timeout: int = 120
    cookies: Dict[str, str] = field(default_factory=dict)


class ConfigManager:
    """Thread-safe configuration manager."""

    def __init__(self, config_file: str = "config.json") -> None:
        self.config_file = Path(config_file)
        self.cookies_file = self.config_file.parent / "cookies.json"
        self.lock = threading.RLock()
        self.config: AppConfig = AppConfig()
        self.load()

    def load(self) -> None:
        with self.lock:
            if self.config_file.exists():
                data = json.loads(self.config_file.read_text(encoding="utf-8"))
                self.config = AppConfig(
                    api_keys=data.get("api_keys", self.config.api_keys),
                    host=data.get("host", self.config.host),
                    port=data.get("port", self.config.port),
                    base_url=data.get("base_url", self.config.base_url),
                    default_model=data.get("default_model", self.config.default_model),
                    request_timeout=data.get("request_timeout", self.config.request_timeout),
                    cookies=data.get("cookies", self.config.cookies),
                )
            # Load persisted cookies from cookies.json (merge with config cookies)
            self._load_cookies_from_disk()
            # Override with environment variables
            self.config.host = os.getenv("HOST", self.config.host)
            self.config.port = int(os.getenv("PORT", str(self.config.port)))
            env_keys = os.getenv("API_KEYS")
            if env_keys:
                self.config.api_keys = [k.strip() for k in env_keys.split(",") if k.strip()]
            env_model = os.getenv("DEFAULT_MODEL")
            if env_model:
                self.config.default_model = env_model

    def _load_cookies_from_disk(self) -> None:
        """Load cookies from cookies.json on disk."""
        if self.cookies_file.exists():
            try:
                data = json.loads(self.cookies_file.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data:
                    self.config.cookies.update(data)
            except (json.JSONDecodeError, OSError):
                pass

    def save_cookies(self) -> None:
        """Persist current cookies to cookies.json on disk."""
        with self.lock:
            try:
                self.cookies_file.write_text(
                    json.dumps(self.config.cookies, indent=2),
                    encoding="utf-8",
                )
            except OSError:
                pass

    def set_cookies(self, cookies: Dict[str, str]) -> None:
        """Update cookies and persist to disk."""
        with self.lock:
            self.config.cookies.update(cookies)
        self.save_cookies()

    def clear_cookies(self) -> None:
        """Clear all stored cookies and remove cookies.json."""
        with self.lock:
            self.config.cookies.clear()
        try:
            if self.cookies_file.exists():
                self.cookies_file.unlink()
        except OSError:
            pass

    def is_valid_cookie(self) -> bool:
        """Check if cookies exist and are not empty."""
        with self.lock:
            return bool(self.config.cookies)

    def is_valid_api_key(self, key: str) -> bool:
        if not key:
            return False
        with self.lock:
            return key in self.config.api_keys