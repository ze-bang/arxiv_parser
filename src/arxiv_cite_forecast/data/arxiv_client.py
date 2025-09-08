from __future__ import annotations

import datetime as dt
import time
from typing import Dict, Iterable, List, Optional

import httpx
from bs4 import BeautifulSoup

from ..utils import Cache, stable_hash, user_agent


ARXIV_BASE = "https://export.arxiv.org/api/query"


class ArxivClient:
    def __init__(self, rate_limit_per_sec: float = 1.0):
        self._cache = Cache("arxiv")
        self._client = httpx.Client(headers={"User-Agent": user_agent()}, timeout=30)
        self._min_interval = 1.0 / max(rate_limit_per_sec, 0.1)
        self._last_call = 0.0

    def _throttle(self):
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    def search(self, query: str, start: int = 0, max_results: int = 100) -> List[dict]:
        params = {
            "search_query": query,
            "start": str(start),
            "max_results": str(max_results),
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
        key = stable_hash({"endpoint": "search", **params})
        cached = self._cache.get(key)
        if cached:
            return cached["entries"]

        self._throttle()
        r = self._client.get(ARXIV_BASE, params=params)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "xml")
        out: List[dict] = []
        for entry in soup.find_all("entry"):
            arxiv_id = entry.id.text.split("/abs/")[-1]
            title = (entry.title.text or "").strip().replace("\n", " ")
            abstract = (entry.summary.text or "").strip().replace("\n", " ")
            categories = [t.get("term") for t in entry.find_all("category") if t.get("term")]
            authors = [a.find("name").text for a in entry.find_all("author") if a.find("name")]
            published = entry.published.text if entry.published else None
            updated = entry.updated.text if entry.updated else None
            out.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                    "categories": categories,
                    "authors": authors,
                    "published": published,
                    "updated": updated,
                }
            )

        self._cache.set(key, {"entries": out})
        return out

    def get_by_id(self, arxiv_id: str) -> Optional[dict]:
        # arXiv API supports id_list, but for simplicity reuse search and filter
        results = self.search(f"id:{arxiv_id}", start=0, max_results=1)
        if results:
            return results[0]
        return None

    def iter_category(
        self,
        category: str,
        start_date: str,
        end_date: str,
        batch_size: int = 100,
        limit: Optional[int] = None,
    ) -> Iterable[dict]:
        start_dt = dt.datetime.fromisoformat(start_date)
        end_dt = dt.datetime.fromisoformat(end_date)
        fetched = 0
        start = 0
        while True:
            query = f"cat:{category}"
            entries = self.search(query, start=start, max_results=batch_size)
            if not entries:
                break
            for e in entries:
                pub = e.get("published")
                if not pub:
                    continue
                try:
                    pub_dt = dt.datetime.fromisoformat(pub.replace("Z", "+00:00")).replace(tzinfo=None)
                except Exception:
                    continue
                if pub_dt < start_dt:
                    continue
                if pub_dt > end_dt:
                    return
                yield e
                fetched += 1
                if limit and fetched >= limit:
                    return
            start += batch_size
