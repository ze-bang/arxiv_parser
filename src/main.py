from __future__ import annotations

import argparse
import datetime as dt
import logging
from typing import List, Optional

from .config import config
from .db import init_db, add_subscriber, remove_subscriber, list_subscribers, mark_seen, filter_unseen
from .arxiv_client import fetch_daily_submissions
from .openalex_client import (
    search_author,
    get_author_metrics,
    get_venue_metrics_from_doi,
    search_venue_by_name,
    get_venue_metrics,
)
from .scoring import PaperSignals, compute_score, explain_score, compute_paper_topic_activity_score
from .summarizer import summarize_with_llm
from .mailer import render_digest, send_email


logger = logging.getLogger("arxiv_parser")


def run(top_m: int, send_mail: bool, dry_run: bool, date: dt.date | None) -> int:
    init_db()

    try:
        UTC = dt.UTC
    except AttributeError:  # pragma: no cover
        from datetime import timezone as _tz
        UTC = _tz.utc
    target_date = date or dt.datetime.now(UTC).date()
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
    def _positional_weights(n: int) -> list[float]:
        if n <= 0:
            return []
        if n == 1:
            return [1.0]
        if n == 2:
            return [0.6, 0.4]
        if n == 3:
            return [0.45, 0.25, 0.30]
        # n >= 4
        first, second, last = 0.40, 0.20, 0.25
        middle_total = 1.0 - (first + second + last)  # 0.15
        mids = n - 3
        per_mid = middle_total / mids
        w = [0.0] * n
        w[0] = first
        w[1] = second
        w[-1] = last
        for i in range(2, n - 1):
            w[i] = per_mid
        return w

    for e in entries:
        names = e.authors
        n = len(names)
        # Choose which authors to query: always include first, second (if exists), and last; fill with early middles up to cap
        CAP = 10
        idxs: list[int] = []
        if n >= 1:
            idxs.append(0)
        if n >= 2:
            idxs.append(1)
        if n >= 3:
            if (n - 1) not in idxs:
                idxs.append(n - 1)
        # Fill remaining slots with 2..n-2
        for i in range(2, max(2, n - 1)):
            if len(idxs) >= CAP:
                break
            if i not in idxs and i < n - 1:
                idxs.append(i)

        # Fetch metrics for selected indices
        per_author_h: list[Optional[float]] = [None] * n
        per_author_w5y: list[Optional[float]] = [None] * n
        for i in idxs:
            a = names[i]
            logger.debug("[OpenAlex] Searching author@%d: %s", i, a)
            if a not in author_cache:
                aid = search_author(a)
                if aid:
                    logger.debug("[OpenAlex] Author found: %s -> %s", a, aid)
                if aid:
                    try:
                        mets = get_author_metrics(aid)
                        author_cache[a] = (aid, {"h": mets.h_index, "w5y": mets.works_count_5y})
                        logger.debug(
                            "[OpenAlex] Metrics for %s (id=%s): h_index=%s, works_5y=%s",
                            a,
                            aid,
                            mets.h_index,
                            mets.works_count_5y,
                        )
                    except Exception:
                        logger.debug("[OpenAlex] Failed to fetch metrics for %s (id=%s)", a, aid, exc_info=True)
                        author_cache[a] = (aid, {"h": None, "w5y": None})
                else:
                    logger.debug("[OpenAlex] No author match for: %s", a)
                    author_cache[a] = (None, {"h": None, "w5y": None})
            _, m = author_cache[a]
            per_author_h[i] = float(m["h"]) if m.get("h") is not None else None
            per_author_w5y[i] = float(m["w5y"]) if m.get("w5y") is not None else None

        weights = _positional_weights(n)
        # Renormalize weights over authors with available metric values
        def _weighted_avg(values: list[Optional[float]]) -> Optional[float]:
            pairs = [(v, weights[i]) for i, v in enumerate(values) if v is not None and i < len(weights)]
            if not pairs:
                return None
            wsum = sum(w for _, w in pairs)
            if wsum <= 0:
                return None
            return sum(v * w for v, w in pairs) / wsum

        h_avg = _weighted_avg(per_author_h)
        w_avg = _weighted_avg(per_author_w5y)
        logger.debug("[Score] Author weights=%s | h_avg=%.3f | w5y_avg=%.3f", weights, h_avg or -1.0, w_avg or -1.0)

        # Topic activity score using LLM-generated keywords
        topic_activity_score = 0.0
        topic_details = {}
        try:
            # Use new LLM-based keyword extraction and scoring
            topic_activity_score, topic_details = compute_paper_topic_activity_score(e.title, e.summary)
            logger.debug("[Score] Topic activity score=%.3f (LLM keywords)", topic_activity_score)
        except Exception as ex:
            logger.warning("[Score] Topic activity scoring failed: %s", ex)
            topic_activity_score = 0.0
            topic_details = {}

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
            venue_two_year_mean_citedness=venue_metric,
            published=e.published,
            topic_activity_score=topic_activity_score,
        )
        score = compute_score(sig, e.title, e.summary, e.authors, topic_details)
        expl = explain_score(sig, score, topic_details)
        # Always include topic activity score since it's now required
        expl["topic_activity_score"] = topic_activity_score

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
                "explanation": expl,
                "topic_activity_score": topic_activity_score,
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
    p.add_argument("--debug", action="store_true", help="Enable debug logging (OpenAlex lookups, etc.)")
    p.add_argument("--dry-run", action="store_true", help="Do not send emails or mark seen")

    sp_add = sub.add_parser("subscribe", help="Add a subscriber email")
    sp_add.add_argument("--email", required=True)

    sp_rm = sub.add_parser("unsubscribe", help="Remove a subscriber email")
    sp_rm.add_argument("--email", required=True)

    args = p.parse_args()

    init_db()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

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
