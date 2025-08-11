from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class PaperSignals:
    author_h_index_avg: float | None
    author_recent_works_avg: float | None
    topic_high_cited_3y: int | None
    venue_two_year_mean_citedness: float | None
    published: dt.datetime


@dataclass
class PaperScore:
    raw: float
    components: dict[str, float]


TAU_DAYS = 365 * 1.5


def time_decay(published: dt.datetime, now: Optional[dt.datetime] = None) -> float:
    now = now or dt.datetime.utcnow()
    age_days = (now - published).days
    return math.exp(-age_days / TAU_DAYS)


def z(x: Optional[float], mean: float, std: float, cap_low: float | None = None, cap_high: float | None = None) -> float:
    if x is None:
        return 0.0
    if std <= 0:
        v = x - mean
    else:
        v = (x - mean) / std
    if cap_low is not None:
        v = max(v, cap_low)
    if cap_high is not None:
        v = min(v, cap_high)
    return v


DEFAULT_STATS = {
    "h_index": (20.0, 15.0),
    "works_5y": (10.0, 8.0),
    "topic_high_cited_3y": (200.0, 250.0),
    "venue_2y_mean_citedness": (1.0, 0.8),
}


def compute_score(sig: PaperSignals) -> PaperScore:
    # Normalize components with rough priors, then weighted sum
    h_m, h_s = DEFAULT_STATS["h_index"]
    w_m, w_s = DEFAULT_STATS["works_5y"]
    t_m, t_s = DEFAULT_STATS["topic_high_cited_3y"]
    v_m, v_s = DEFAULT_STATS["venue_2y_mean_citedness"]

    comps = {
        "author_h_index": z(sig.author_h_index_avg, h_m, h_s, -2, 4),
        "author_recent": z(sig.author_recent_works_avg, w_m, w_s, -2, 4),
        "topic": z(sig.topic_high_cited_3y, t_m, t_s, -2, 4),
        "venue": z(sig.venue_two_year_mean_citedness, v_m, v_s, -2, 4),
    }

    # Weights can be tuned
    weights = {"author_h_index": 0.35, "author_recent": 0.25, "topic": 0.25, "venue": 0.15}

    base = sum(comps[k] * w for k, w in weights.items())
    decay = time_decay(sig.published)
    score = base * decay
    comps["time_decay"] = decay

    return PaperScore(raw=score, components=comps)
