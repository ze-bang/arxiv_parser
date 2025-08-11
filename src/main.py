from __future__ import annotations

import argparse
import datetime as dt
from typing import List

from .config import config
from .db import init_db, add_subscriber, remove_subscriber, list_subscribers, mark_seen, filter_unseen
from .arxiv_client import fetch_daily_submissions
from .openalex_client import (
    search_author,
    get_author_metrics,
    search_concept,
    get_topic_metrics,
    get_venue_metrics_from_doi,
    search_venue_by_name,
    get_venue_metrics,
)
from .scoring import PaperSignals, compute_score
from .summarizer import summarize_with_llm
from .mailer import render_digest, send_email


def _infer_concept_query(title: str, categories: list[str]) -> str:
    # Simple heuristic: use title; categories narrowed already to cond-mat.str-el
    return title


def run(top_m: int, send_mail: bool, dry_run: bool, date: dt.date | None) -> int:
    init_db()

    target_date = date or dt.datetime.utcnow().date()
    entries = fetch_daily_submissions(config.category, target_date)
    if not entries:
        if dry_run:
            print(f"No submissions for {target_date}.")
        if send_mail:
            subs = get_subscribers()
            if subs:
                html = render_digest(target_date.isoformat(), [])
                send_email(f"cond-mat.str-el Daily Digest ({target_date}, no submissions)", html, subs)
        return 0

    # Resolve author metrics with simple caching in-memory
    author_cache: dict[str, tuple[str | None, dict]] = {}

    items = []
    ids = [e.id for e in entries]
    title_by_id = {e.id: e.title for e in entries}
    published_by_id = {e.id: e.published.date().isoformat() for e in entries}

    # Compute scores
    for e in entries:
        metrics = []
        h_vals = []
        w_vals = []
        for a in e.authors[:8]:  # cap number of authors to query
            if a not in author_cache:
                aid = search_author(a)
                if aid:
                    try:
                        mets = get_author_metrics(aid)
                        author_cache[a] = (aid, {"h": mets.h_index, "w5y": mets.works_count_5y})
                    except Exception:
                        author_cache[a] = (aid, {"h": None, "w5y": None})
                else:
                    author_cache[a] = (None, {"h": None, "w5y": None})
            _, m = author_cache[a]
            if m.get("h") is not None:
                h_vals.append(float(m["h"]))
            if m.get("w5y") is not None:
                w_vals.append(float(m["w5y"]))

        h_avg = sum(h_vals) / len(h_vals) if h_vals else None
        w_avg = sum(w_vals) / len(w_vals) if w_vals else None

        # Topic metrics via concept search
        topic_query = _infer_concept_query(e.title, e.categories)
        topic_id = None
        topic_high_cited = None
        try:
            topic_id = search_concept(topic_query)
            if topic_id:
                topic_high_cited = get_topic_metrics(topic_id).high_cited_papers_3y
        except Exception:
            topic_high_cited = None

        # Venue metrics if possible
        venue_metric = None
        if getattr(e, "doi", None):
            try:
                venue_metric = get_venue_metrics_from_doi(e.doi).two_year_mean_citedness
            except Exception:
                venue_metric = None
        if venue_metric is None and getattr(e, "journal_ref", None):
            try:
                v_id = search_venue_by_name(e.journal_ref)
                if v_id:
                    venue_metric = get_venue_metrics(v_id).two_year_mean_citedness
            except Exception:
                pass

        sig = PaperSignals(
            author_h_index_avg=h_avg,
            author_recent_works_avg=w_avg,
            topic_high_cited_3y=topic_high_cited,
            venue_two_year_mean_citedness=venue_metric,
            published=e.published,
        )
        score = compute_score(sig)

        items.append(
            {
                "arxiv_id": e.id,
                "title": e.title,
                "authors": e.authors,
                "published": e.published.date().isoformat(),
                "link": e.link,
                "components": score.components,
                "score": score.raw,
                "abstract": e.summary,
            }
        )

    # Sort and take top-m
    items.sort(key=lambda x: x["score"], reverse=True)
    top_items = items[:top_m]

    # Generate LLM summaries
    for it in top_items:
        it["summary"] = summarize_with_llm(it["title"], it["abstract"], it["components"])  # replaces later in email

    # Prepare email body
    html = render_digest(target_date.isoformat(), top_items)

    if dry_run:
        print(f"Fetched {len(entries)} submissions. Top {len(top_items)} prepared.")
        print("Subscribers:", list_subscribers())
        print(html[:800], "...\n[truncated]")
    else:
        # mark seen
        mark_seen([e.id for e in entries], title_by_id, published_by_id)
        if send_mail:
            subs = get_subscribers()
            if subs:
                send_email(f"cond-mat.str-el Daily Digest ({target_date})", html, subs)

    return 0


def cli():
    p = argparse.ArgumentParser(description="Daily arXiv cond-mat.str-el digest")
    sub = p.add_subparsers(dest="cmd")

    p.add_argument("--top", type=int, default=config.top_m, help="Top m papers to include")
    p.add_argument("--send-email", action="store_true", help="Send digest to subscribers")
    p.add_argument(
        "--date",
        type=str,
        default=None,
        help="Fetch submissions for YYYY-MM-DD (UTC); defaults to today",
    )
    p.add_argument("--dry-run", action="store_true", help="Do not send emails or mark seen")

    sp_add = sub.add_parser("subscribe", help="Add a subscriber email")
    sp_add.add_argument("--email", required=True)

    sp_rm = sub.add_parser("unsubscribe", help="Remove a subscriber email")
    sp_rm.add_argument("--email", required=True)

    args = p.parse_args()

    init_db()

    if args.cmd == "subscribe":
        ok = add_subscriber(args.email)
        print("Added" if ok else "Already subscribed")
        return
    if args.cmd == "unsubscribe":
        ok = remove_subscriber(args.email)
        print("Removed" if ok else "Not found")
        return

    top = int(args.top)
    date_arg = None
    if args.date:
        try:
            date_arg = dt.datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print("Invalid --date format; expected YYYY-MM-DD. Using today.")
    exit(run(top, args.send_email, args.dry_run, date_arg))


# Remote subscribers support (Vercel API)
def get_subscribers() -> list[str]:
    if config.subscribers_url:
        try:
            import requests  # lazy import
            r = requests.get(config.subscribers_url, timeout=15)
            r.raise_for_status()
            j = r.json()
            # expected shape: { subscribers: ["a@b.com", ...] }
            subs = j.get("subscribers")
            if isinstance(subs, list):
                return [s for s in subs if isinstance(s, str)]
        except Exception:
            pass
    return list_subscribers()


if __name__ == "__main__":
    cli()
