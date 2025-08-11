from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import List, Optional
import feedparser


@dataclass
class ArxivEntry:
    id: str
    title: str
    summary: str
    authors: list[str]
    published: dt.datetime
    categories: list[str]
    link: str
    doi: str | None
    journal_ref: str | None


def parse_arxiv_id(entry_id: str) -> str:
    # e.g., http://arxiv.org/abs/2401.01234v1 -> 2401.01234v1
    return entry_id.rsplit("/", 1)[-1]


def fetch_submissions_for_date(category: str, date: dt.date) -> List[ArxivEntry]:
    # arXiv API supports date range via search_query=cat:... AND submittedDate:[YYYYMMDD0000 TO YYYYMMDD2359]
    yyyymmdd = date.strftime("%Y%m%d")
    query = f"cat:{category}+AND+submittedDate:[{yyyymmdd}0000+TO+{yyyymmdd}2359]"
    url = (
        "http://export.arxiv.org/api/query?search_query="
        f"{query}&start=0&max_results=200&sortBy=submittedDate&sortOrder=ascending"
    )
    feed = feedparser.parse(url)
    results: List[ArxivEntry] = []
    # Determine UTC constant compatibly
    try:
        UTC = dt.UTC  # Python 3.11+
    except AttributeError:  # pragma: no cover
        from datetime import timezone as _tz
        UTC = _tz.utc
    for e in feed.entries:
        categories = [t["term"] for t in getattr(e, "tags", []) if "term" in t]
        if getattr(e, "published_parsed", None):
            published = dt.datetime(*e.published_parsed[:6]).replace(tzinfo=UTC)
        else:
            published = dt.datetime.now(UTC)
        # arXiv feedparser exposes extra fields under the 'arxiv_' namespace when present
        doi = getattr(e, "arxiv_doi", None)
        journal_ref = getattr(e, "arxiv_journal_ref", None)
        results.append(
            ArxivEntry(
                id=parse_arxiv_id(e.id),
                title=e.title.strip(),
                summary=e.summary.strip(),
                authors=[a.name for a in e.authors] if getattr(e, "authors", None) else [],
                published=published,
                categories=categories,
                link=e.link,
                doi=doi,
                journal_ref=journal_ref,
            )
        )
    return results


def fetch_daily_submissions(category: str, date: Optional[dt.date] = None) -> List[ArxivEntry]:
    """Backward-compatible wrapper to fetch submissions for a given date.
    If date is None, uses today's UTC date.
    """
    try:
        UTC = dt.UTC
    except AttributeError:  # pragma: no cover
        from datetime import timezone as _tz
        UTC = _tz.utc
    d = date or dt.datetime.now(UTC).date()
    return fetch_submissions_for_date(category, d)
