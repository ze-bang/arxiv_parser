from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional
import requests
import logging
import datetime as dt

BASE = "https://api.openalex.org"
HEADERS = {"User-Agent": "arxiv_parser/0.1", "Accept": "application/json"}


@dataclass
class AuthorMetrics:
    h_index: int | None
    works_count_5y: int | None


@dataclass
class VenueMetrics:
    two_year_mean_citedness: float | None


@dataclass
class TopicMetrics:
    high_cited_papers_3y: int | None


def _get(url: str, params: Optional[dict] = None, retry: int = 3):
    logger = logging.getLogger("arxiv_parser")
    for i in range(retry):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        except Exception as e:
            logger.debug("[OpenAlex] GET failed: %s params=%s err=%s", url, params, e)
            raise
        logger.debug("[OpenAlex] GET %s params=%s -> %s", r.url, params, r.status_code)
        if r.status_code == 200:
            return r.json()
        time.sleep(1 + i)
    r.raise_for_status()


def _entity_code(id_or_url: str) -> str:
    # Extract trailing code like A..., V..., C..., W...
    return id_or_url.rsplit("/", 1)[-1]


def _entity_resource_for_code(code: str) -> Optional[str]:
    m = {
        "A": "authors",
        "V": "venues",
        "C": "concepts",
        "W": "works",
        "I": "institutions",
        "S": "sources",
    }
    return m.get(code[:1].upper())


def _to_api_entity_url(id_or_url: str) -> str:
    # If already an API URL, return as is; otherwise convert openalex.org/Code to api.openalex.org/<resource>/Code
    if id_or_url.startswith(BASE + "/"):
        return id_or_url
    code = _entity_code(id_or_url)
    res = _entity_resource_for_code(code)
    if res:
        return f"{BASE}/{res}/{code}"
    return id_or_url.replace("https://openalex.org", BASE).replace("http://openalex.org", BASE)


def search_author(name: str) -> Optional[str]:
    j = _get(f"{BASE}/authors", params={"search": name, "per_page": 1})
    return _to_api_entity_url(j["results"][0]["id"]) if j.get("results") else None


def get_author_metrics(author_id: str) -> AuthorMetrics:
    j = _get(_to_api_entity_url(author_id))
    h = j.get("summary_stats", {}).get("h_index")
    # recent works count (last 5y)
    wc = j.get("counts_by_year", [])
    works_5y = sum(y.get("works_count", 0) for y in wc if y.get("year", 0) >= (wc[0]["year"] - 4) if wc)
    return AuthorMetrics(h_index=h, works_count_5y=works_5y)


def search_venue_by_issn(issn: str) -> Optional[str]:
    j = _get(f"{BASE}/venues", params={"search": issn, "per_page": 1})
    return _to_api_entity_url(j["results"][0]["id"]) if j.get("results") else None


def get_venue_metrics(venue_id: str) -> VenueMetrics:
    j = _get(_to_api_entity_url(venue_id))
    tymc = j.get("summary_stats", {}).get("2yr_mean_citedness")
    return VenueMetrics(two_year_mean_citedness=tymc)


def get_venue_metrics_from_doi(doi: str) -> VenueMetrics:
    # Look up work by DOI, then derive venue and metrics
    w = _get(f"{BASE}/works/https://doi.org/{doi}")
    venue = w.get("host_venue", {})
    venue_id = venue.get("id")
    if venue_id:
        return get_venue_metrics(_to_api_entity_url(venue_id))
    return VenueMetrics(two_year_mean_citedness=None)


def search_venue_by_name(name: str) -> Optional[str]:
    j = _get(f"{BASE}/venues", params={"search": name, "per_page": 1})
    return _to_api_entity_url(j["results"][0]["id"]) if j.get("results") else None


def search_concept(query: str) -> Optional[str]:
    j = _get(f"{BASE}/concepts", params={"search": query, "per_page": 1})
    if j.get("results"):
        return j["results"][0]["id"]
    return None


def get_topic_metrics(concept_id: str) -> TopicMetrics:
    # Highly cited works in last 3y: filter works with cited_by_count >= 50 and from last 3 years
    import datetime as dt
    try:
        UTC = dt.UTC
    except AttributeError:  # pragma: no cover
        from datetime import timezone as _tz
        UTC = _tz.utc
    year = dt.datetime.now(UTC).year
    code = _entity_code(concept_id)
    params = {
        "filter": f"from_publication_date:{year-2}-01-01,concepts.id:{code},cited_by_count:>50",
        "per_page": 1,
    }
    j = _get(f"{BASE}/works", params=params)
    # Count is in meta -> count
    count = j.get("meta", {}).get("count")
    return TopicMetrics(high_cited_papers_3y=count)


# --- New helpers for topic activity scoring ---
def search_work_by_title(title: str) -> Optional[dict]:
    """Return the first matching work object for a title search (includes id, concepts, host_venue, etc.)."""
    j = _get(f"{BASE}/works", params={"search": title, "per_page": 1})
    res = j.get("results") or []
    return res[0] if res else None


def extract_concepts_from_work(work: dict, top_n: int = 5) -> list[dict]:
    """Extract top concepts from a Work result; returns list of dicts with id, display_name, score."""
    concepts = work.get("concepts") or []
    # Sort by provided score descending if present
    concepts.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    return concepts[:top_n]


def list_recent_works_for_concept(concept_id: str, from_date: str, per_page: int = 25, max_records: int = 100) -> list[dict]:
    """List recent works for a concept since from_date (YYYY-MM-DD). Limits to max_records for performance."""
    code = _entity_code(concept_id)
    results: list[dict] = []
    page = 1
    while len(results) < max_records:
        params = {
            "filter": f"from_publication_date:{from_date},concepts.id:{code}",
            "per_page": per_page,
            "page": page,
            "sort": "publication_date:desc",
        }
        j = _get(f"{BASE}/works", params=params)
        batch = j.get("results") or []
        if not batch:
            break
        results.extend(batch)
        if len(batch) < per_page:
            break
        page += 1
    return results[:max_records]


def parse_publication_date(work: dict) -> dt.date | None:
    d = work.get("publication_date") or work.get("from_publication_date") or None
    if d:
        try:
            return dt.date.fromisoformat(d)
        except Exception:
            pass
    y = work.get("publication_year")
    if isinstance(y, int) and y > 0:
        return dt.date(y, 6, 30)
    return None
