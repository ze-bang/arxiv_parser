from __future__ import annotations

import datetime as dt
import math
import logging
import random
from dataclasses import dataclass
from typing import Optional
from typing import Iterable

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None  # type: ignore

from .config import config


@dataclass
class PaperSignals:
    author_h_index_avg: float | None
    author_recent_works_avg: float | None
    venue_two_year_mean_citedness: float | None
    published: dt.datetime
    # Required: LLM-generated keywords-based topic activity score
    topic_activity_score: float


@dataclass
class PaperScore:
    raw: float
    components: dict[str, float]


TAU_DAYS = 365 * 1.5


def time_decay(published: dt.datetime, now: Optional[dt.datetime] = None) -> float:
    try:
        UTC = dt.UTC
    except AttributeError:  # pragma: no cover
        from datetime import timezone as _tz
        UTC = _tz.utc
    now = now or dt.datetime.now(UTC)
    # Ensure both datetimes are comparable (timezone-aware)
    if published.tzinfo is None:
        published = published.replace(tzinfo=UTC)
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
    "venue_2y_mean_citedness": (1.0, 0.8),
    # For activity-based topic score we use log1p transformation; priors below are for log1p(activity)
    "topic_activity_log1p": (5.0, 2.0),  # Updated for more sophisticated LLM-based scoring
}


# Public weights used for combining component z-scores
WEIGHTS: dict[str, float] = {
    "author_h_index": 0.35,
    "author_recent": 0.25,
    "topic": 0.25,
    "venue": 0.15,
}


def compute_score(sig: PaperSignals) -> PaperScore:
    # Normalize components with rough priors, then weighted sum
    h_m, h_s = DEFAULT_STATS["h_index"]
    w_m, w_s = DEFAULT_STATS["works_5y"]
    # Use only the LLM-based keyword topic activity score
    t_m, t_s = DEFAULT_STATS["topic_activity_log1p"]
    topic_signal = math.log1p(max(sig.topic_activity_score, 0.0))
    v_m, v_s = DEFAULT_STATS["venue_2y_mean_citedness"]

    comps = {
        "author_h_index": z(sig.author_h_index_avg, h_m, h_s, -2, 4),
        "author_recent": z(sig.author_recent_works_avg, w_m, w_s, -2, 4),
        "topic": z(topic_signal, t_m, t_s, -2, 4),
        "venue": z(sig.venue_two_year_mean_citedness, v_m, v_s, -2, 4),
    }

    # Weights can be tuned
    weights = WEIGHTS

    base = sum(comps[k] * w for k, w in weights.items())
    decay = time_decay(sig.published)
    score = base * decay
    comps["time_decay"] = decay

    return PaperScore(raw=score, components=comps)


# --- Topic activity scoring with LLM-generated keywords and log-normal projection ---

def extract_keywords_with_llm(title: str, abstract: str, num_keywords: int = 5) -> list[str]:
    """Use LLM to extract the most relevant keywords for topic search."""
    if not config.openai_api_key or OpenAI is None:
        # Fallback to basic keyword extraction from title/abstract
        logger = logging.getLogger("arxiv_parser")
        logger.warning("OpenAI not configured; using basic keyword extraction")
        # Simple fallback: extract capitalized words and common physics terms
        words = (title + " " + abstract).split()
        keywords = []
        physics_terms = ["quantum", "condensed", "matter", "magnetic", "electronic", "phase", "transition", "superconducting"]
        for word in words:
            clean_word = word.strip(".,;:()[]").lower()
            if (word[0].isupper() and len(word) > 3) or clean_word in physics_terms:
                if clean_word not in keywords:
                    keywords.append(clean_word)
                if len(keywords) >= num_keywords:
                    break
        return keywords[:num_keywords]

    client = OpenAI(api_key=config.openai_api_key)
    prompt = (
        f"Extract {num_keywords} most relevant scientific keywords for this paper that would be useful "
        f"for finding similar research articles. Focus on specific technical terms, materials, methods, "
        f"or phenomena. Return only the keywords separated by commas.\n\n"
        f"Title: {title}\n\nAbstract: {abstract}"
    )
    
    try:
        resp = client.chat.completions.create(
            model=config.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100
        )
        keywords_text = resp.choices[0].message.content.strip()
        keywords = [kw.strip().lower() for kw in keywords_text.split(",")]
        return keywords[:num_keywords]
    except Exception as e:
        logger = logging.getLogger("arxiv_parser")
        logger.warning("LLM keyword extraction failed: %s", e)
        return []


def log_normal_projected_citations(journal_impact_factor: float, years_since_publication: float) -> float:
    """
    Generate projected total citations using log-normal distribution.
    
    Args:
        journal_impact_factor: The impact factor of the journal (mean of the distribution)
        years_since_publication: Years since publication for time-based scaling
    
    Returns:
        Projected total citations
    """
    if journal_impact_factor <= 0:
        return 0.0
    
    # Log-normal parameters: mean impact factor, some standard deviation
    mu = math.log(max(journal_impact_factor, 0.1))  # log of the mean
    sigma = 0.8  # Standard deviation in log space (tunable parameter)
    
    # Generate log-normal sample
    # Using Box-Muller transform for normal distribution, then exp for log-normal
    u1, u2 = random.random(), random.random()
    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    log_normal_sample = math.exp(mu + sigma * z)
    
    # Scale by time factor (papers accumulate more citations over time)
    # Assume citations grow with square root of time (diminishing returns)
    time_factor = max(1.0, math.sqrt(max(years_since_publication, 0.1)))
    
    return log_normal_sample * time_factor


def projected_citations_for_work(work: dict) -> float:
    """
    Calculate projected citations for a work based on its journal's impact factor.
    
    Args:
        work: OpenAlex work dict containing host_venue and publication info
    
    Returns:
        Projected citations count
    """
    # Check if we already have actual citations
    actual_citations = work.get("cited_by_count")
    if isinstance(actual_citations, int) and actual_citations > 0:
        return float(actual_citations)
    
    # Get journal impact factor from host venue
    host_venue = work.get("host_venue") or {}
    # Use venue's 2-year mean citedness as proxy for impact factor
    impact_factor = 1.0  # Default
    
    # Try to get impact factor from venue stats (if available in the work data)
    venue_stats = host_venue.get("summary_stats", {})
    if venue_stats:
        impact_factor = venue_stats.get("2yr_mean_citedness", 1.0)
    
    # Calculate years since publication
    pub_date = publication_datetime_from_work(work)
    try:
        UTC = dt.UTC
    except AttributeError:
        from datetime import timezone as _tz
        UTC = _tz.utc
    now = dt.datetime.now(UTC)
    years_since = max(0.1, (now - pub_date).days / 365.25)
    
    return log_normal_projected_citations(impact_factor, years_since)


def compute_topic_activity_score_with_keywords(keywords: list[str], max_works_per_keyword: int = 50) -> float:
    """
    Compute topic activity score using LLM-generated keywords.
    
    For each keyword:
    1. Search for recent articles containing the keyword
    2. For each article found, calculate: projected_citations * time_decay
    3. Sum all contributions
    
    Args:
        keywords: List of relevant keywords from LLM
        max_works_per_keyword: Maximum number of works to consider per keyword
    
    Returns:
        Total topic activity score
    """
    from . import openalex_client
    
    total_score = 0.0
    logger = logging.getLogger("arxiv_parser")
    
    # Search from 3 years ago to capture recent activity
    try:
        UTC = dt.UTC
    except AttributeError:
        from datetime import timezone as _tz
        UTC = _tz.utc
    
    from_date = (dt.datetime.now(UTC) - dt.timedelta(days=3*365)).strftime("%Y-%m-%d")
    
    for keyword in keywords:
        try:
            # Search for works containing this keyword
            works = search_works_by_keyword(keyword, from_date, max_works_per_keyword)
            logger.debug(f"Found {len(works)} works for keyword '{keyword}'")
            
            for work in works:
                projected_cites = projected_citations_for_work(work)
                pub_datetime = publication_datetime_from_work(work)
                decay = time_decay(pub_datetime)
                contribution = projected_cites * decay
                total_score += contribution
                
        except Exception as e:
            logger.warning(f"Error processing keyword '{keyword}': {e}")
            continue
    
    return total_score


def search_works_by_keyword(keyword: str, from_date: str, max_results: int = 50) -> list[dict]:
    """
    Search OpenAlex for works containing a specific keyword.
    
    Args:
        keyword: The keyword to search for
        from_date: ISO date string (YYYY-MM-DD) for filtering recent works
        max_results: Maximum number of results to return
    
    Returns:
        List of work dictionaries from OpenAlex
    """
    from . import openalex_client
    
    # Use OpenAlex search with keyword filter
    params = {
        "search": keyword,
        "filter": f"from_publication_date:{from_date}",
        "per_page": min(max_results, 100),
        "sort": "publication_date:desc"
    }
    
    try:
        BASE = "https://api.openalex.org"
        HEADERS = {"User-Agent": "arxiv_parser/0.1", "Accept": "application/json"}
        import requests
        
        response = requests.get(f"{BASE}/works", params=params, headers=HEADERS, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        return data.get("results", [])[:max_results]
        
    except Exception as e:
        logger = logging.getLogger("arxiv_parser")
        logger.error(f"Failed to search works for keyword '{keyword}': {e}")
        return []


def publication_datetime_from_work(work: dict) -> dt.datetime:
    d = work.get("publication_date")
    try:
        UTC = dt.UTC
    except AttributeError:  # pragma: no cover
        from datetime import timezone as _tz
        UTC = _tz.utc
    if d:
        try:
            return dt.datetime.fromisoformat(d).replace(tzinfo=UTC)
        except Exception:
            pass
    y = work.get("publication_year")
    if isinstance(y, int) and y > 0:
        return dt.datetime(y, 6, 30, tzinfo=UTC)
    return dt.datetime.now(UTC)


def compute_topic_activity_score(works: Iterable[dict]) -> float:
    """Sum over recent works: projected_citations * time_decay(per work)."""
    total = 0.0
    for w in works:
        projected_cites = projected_citations_for_work(w)
        td = time_decay(publication_datetime_from_work(w))
        total += projected_cites * td
    return total


def explain_score(sig: PaperSignals) -> dict:
    """Return a pedantic breakdown of the score calculation for templating.

    Structure:
    {
      'signals': {... raw values ...},
      'priors': DEFAULT_STATS,
      'z': {... z-scores per component ...},
      'weights': WEIGHTS,
      'contributions': {... z * weight ...},
      'base_sum': float,
      'time_decay': float,
      'final_score': float,
    }
    """
    h_m, h_s = DEFAULT_STATS["h_index"]
    w_m, w_s = DEFAULT_STATS["works_5y"]
    # Use only the LLM-based keyword topic activity score
    t_m, t_s = DEFAULT_STATS["topic_activity_log1p"]
    topic_signal = math.log1p(max(sig.topic_activity_score, 0.0))
    v_m, v_s = DEFAULT_STATS["venue_2y_mean_citedness"]

    z_components = {
        "author_h_index": z(sig.author_h_index_avg, h_m, h_s, -2, 4),
        "author_recent": z(sig.author_recent_works_avg, w_m, w_s, -2, 4),
        "topic": z(topic_signal, t_m, t_s, -2, 4),
        "venue": z(sig.venue_two_year_mean_citedness, v_m, v_s, -2, 4),
    }

    contributions = {k: z_components[k] * WEIGHTS[k] for k in WEIGHTS.keys()}
    base_sum = sum(contributions.values())
    decay = time_decay(sig.published)
    final = base_sum * decay

    return {
        "signals": {
            "author_h_index_avg": sig.author_h_index_avg,
            "author_recent_works_avg": sig.author_recent_works_avg,
            "venue_two_year_mean_citedness": sig.venue_two_year_mean_citedness,
            "published": sig.published.isoformat(),
            "topic_activity_score": sig.topic_activity_score,
            "topic_source": "llm_keywords",
            "topic_signal_used": topic_signal,
        },
        "priors": DEFAULT_STATS,
        "z": z_components,
        "weights": WEIGHTS,
        "contributions": contributions,
        "base_sum": base_sum,
        "time_decay": decay,
        "final_score": final,
    }


def compute_paper_topic_activity_score(title: str, abstract: str, num_keywords: int = 5, max_works_per_keyword: int = 50) -> float:
    """
    Main function to compute topic activity score for a paper using LLM-generated keywords.
    
    Args:
        title: Paper title
        abstract: Paper abstract
        num_keywords: Number of keywords to extract with LLM
        max_works_per_keyword: Maximum works to analyze per keyword
    
    Returns:
        Topic activity score based on projected citations and time decay
    """
    # Step 1: Extract keywords using LLM
    keywords = extract_keywords_with_llm(title, abstract, num_keywords)
    
    if not keywords:
        logging.getLogger("arxiv_parser").warning("No keywords extracted, returning zero topic score")
        return 0.0
    
    # Step 2: Compute activity score using these keywords
    return compute_topic_activity_score_with_keywords(keywords, max_works_per_keyword)
