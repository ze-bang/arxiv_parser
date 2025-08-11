from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional
import requests

BASE = "https://api.openalex.org"
HEADERS = {"User-Agent": "arxiv_parser/0.1"}


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
    for i in range(retry):
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            return r.json()
        time.sleep(1 + i)
    r.raise_for_status()


def search_author(name: str) -> Optional[str]:
    j = _get(f"{BASE}/authors", params={"search": name, "per_page": 1})
    if j.get("results"):
        return j["results"][0]["id"]
    return None


def get_author_metrics(author_id: str) -> AuthorMetrics:
    j = _get(author_id)
    h = j.get("summary_stats", {}).get("h_index")
    # recent works count (last 5y)
    wc = j.get("counts_by_year", [])
    works_5y = sum(y.get("works_count", 0) for y in wc if y.get("year", 0) >= (wc[0]["year"] - 4) if wc)
    return AuthorMetrics(h_index=h, works_count_5y=works_5y)


def search_venue_by_issn(issn: str) -> Optional[str]:
    j = _get(f"{BASE}/venues", params={"search": issn, "per_page": 1})
    if j.get("results"):
        return j["results"][0]["id"]
    return None


def get_venue_metrics(venue_id: str) -> VenueMetrics:
    j = _get(venue_id)
    tymc = j.get("summary_stats", {}).get("2yr_mean_citedness")
    return VenueMetrics(two_year_mean_citedness=tymc)


def get_venue_metrics_from_doi(doi: str) -> VenueMetrics:
    # Look up work by DOI, then derive venue and metrics
    w = _get(f"{BASE}/works/https://doi.org/{doi}")
    venue = w.get("host_venue", {})
    venue_id = venue.get("id")
    if venue_id:
        return get_venue_metrics(venue_id)
    return VenueMetrics(two_year_mean_citedness=None)


def search_venue_by_name(name: str) -> Optional[str]:
    j = _get(f"{BASE}/venues", params={"search": name, "per_page": 1})
    if j.get("results"):
        return j["results"][0]["id"]
    return None


def search_concept(query: str) -> Optional[str]:
    j = _get(f"{BASE}/concepts", params={"search": query, "per_page": 1})
    if j.get("results"):
        return j["results"][0]["id"]
    return None


def get_topic_metrics(concept_id: str) -> TopicMetrics:
    # Highly cited works in last 3y: filter works with cited_by_count >= 50 and from last 3 years
    import datetime as dt

    year = dt.datetime.utcnow().year
    params = {
        "filter": f"from_publication_date:{year-2}-01-01,concepts.id:{concept_id},cited_by_count:>50",
        "per_page": 1,
    }
    j = _get(f"{BASE}/works", params=params)
    # Count is in meta -> count
    count = j.get("meta", {}).get("count")
    return TopicMetrics(high_cited_papers_3y=count)
