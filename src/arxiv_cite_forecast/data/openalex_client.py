from __future__ import annotations

import time
from typing import Optional

import httpx

from ..utils import Cache, stable_hash, user_agent


OPENALEX_BASE = "https://api.openalex.org"


class OpenAlexClient:
    def __init__(self, rate_limit_per_sec: float = 2.0):
        self._cache = Cache("openalex")
        self._client = httpx.Client(headers={"User-Agent": user_agent()}, timeout=30)
        self._min_interval = 1.0 / max(rate_limit_per_sec, 0.1)
        self._last_call = 0.0

    def _throttle(self):
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    def work_by_arxiv_id(self, arxiv_id: str) -> Optional[dict]:
        # Normalize like 2401.00001 -> arXiv:2401.00001
        norm = arxiv_id if arxiv_id.lower().startswith("arxiv:") else f"arXiv:{arxiv_id}"
        key = stable_hash({"endpoint": "work_by_arxiv", "id": norm})
        cached = self._cache.get(key)
        if cached:
            return cached.get("work")

        self._throttle()
        r = self._client.get(f"{OPENALEX_BASE}/works/", params={"search": norm})
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        work = results[0] if results else None
        self._cache.set(key, {"work": work})
        return work

    def work(self, openalex_id: str) -> Optional[dict]:
        key = stable_hash({"endpoint": "work", "id": openalex_id})
        cached = self._cache.get(key)
        if cached:
            return cached.get("work")
        self._throttle()
        r = self._client.get(f"{OPENALEX_BASE}/works/{openalex_id}")
        r.raise_for_status()
        work = r.json()
        self._cache.set(key, {"work": work})
        return work
