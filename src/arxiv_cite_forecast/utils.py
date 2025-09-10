from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv


load_dotenv()


def get_cache_dir() -> Path:
    d = os.getenv("CACHE_DIR", "data_cache")
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_agent() -> str:
    email = os.getenv("CONTACT_EMAIL", "")
    base = "arxiv-citation-forecast/0.1"
    if email:
        return f"{base} (mailto:{email})"
    return base


def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class Cache:
    namespace: str

    def _path(self, key: str) -> Path:
        root = get_cache_dir() / self.namespace
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{key}.json"

    def get(self, key: str) -> Optional[dict]:
        p = self._path(key)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
        return None

    def set(self, key: str, value: dict) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directories exist
        p.write_text(json.dumps(value, ensure_ascii=False))
