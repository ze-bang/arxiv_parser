from __future__ import annotations

import datetime as dt
from typing import Iterable, List, Optional

import pandas as pd

from .arxiv_client import ArxivClient
from .openalex_client import OpenAlexClient


def _compute_12m_citations(work: dict, submission_date: dt.datetime) -> Optional[int]:
    counts = work.get("counts_by_year") or []
    # counts_by_year is list of {year: int, works_count: int?, cited_by_count: int}
    # We'll accumulate citations in the 12 months window post submission.
    # Approximate by calendar years: if submission in 2021-09, take remaining 2021 fraction + part of 2022.
    # For simplicity in demo, we use full next-year if at least 6 months spill; otherwise year of submission.
    if not counts:
        return None
    # Build mapping year->cites
    year2cites = {c.get("year"): c.get("cited_by_count", 0) for c in counts if c.get("year")}
    start_year = submission_date.year
    # Require at least 365 days passed since submission to form label
    if (dt.datetime.utcnow() - submission_date).days < 365:
        return None
    # Simple window: cites in year of submission + following year
    cites = year2cites.get(start_year, 0) + year2cites.get(start_year + 1, 0)
    return int(cites)


def build_dataset(
    category: str,
    start_date: str,
    end_date: str,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    arxiv = ArxivClient()
    oa = OpenAlexClient()

    records: List[dict] = []
    for e in arxiv.iter_category(category, start_date, end_date, batch_size=100, limit=limit):
        arxiv_id = e["arxiv_id"]
        title = e.get("title") or ""
        abstract = e.get("abstract") or ""
        cats = e.get("categories") or []
        authors = e.get("authors") or []
        published = e.get("published")
        try:
            pub_dt = dt.datetime.fromisoformat(published.replace("Z", "+00:00")).replace(tzinfo=None) if published else None
        except Exception:
            pub_dt = None

        work = oa.work_by_arxiv_id(arxiv_id)
        y12 = _compute_12m_citations(work, pub_dt) if (work and pub_dt) else None
        if y12 is None:
            continue  # skip rows without valid target

        records.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "categories": cats,
                "authors": authors,
                "published": pub_dt,
                "openalex_id": (work.get("id") if work else None),
                "y12": y12,
            }
        )

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df["n_authors"] = df["authors"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df["primary_cat"] = df["categories"].apply(lambda x: x[0] if isinstance(x, list) and x else None)
        df["month"] = df["published"].dt.month
    return df
