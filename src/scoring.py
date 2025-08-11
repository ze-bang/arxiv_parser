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


# LLM-based scoring system - no more priors or fixed weights


def _create_topic_summary(topic_details: dict, total_score: float) -> str:
    """Create a curated summary of topic score calculation for LLM."""
    if not topic_details or "keywords" not in topic_details:
        return f"Topic activity score: {total_score:.2f} (no detailed breakdown available)"
    
    keywords = topic_details.get("keywords", [])
    keywords_with_relevance = topic_details.get("keywords_with_relevance", [])
    breakdown = topic_details.get("breakdown", [])
    
    summary_parts = [
        f"Topic activity score: {total_score:.2f}",
        f"Keywords analyzed: {', '.join(keywords)}"
    ]
    
    # Add relevance information if available
    if keywords_with_relevance:
        relevance_info = []
        for kw_data in keywords_with_relevance:
            if isinstance(kw_data, dict):
                keyword = kw_data.get('keyword', '')
                relevance = kw_data.get('relevance', 0.0)
                relevance_info.append(f"'{keyword}' (relevance: {relevance:.1f})")
            else:
                relevance_info.append(f"'{kw_data}' (relevance: unknown)")
        summary_parts.append(f"Keywords with relevance: {', '.join(relevance_info)}")
    
    if breakdown:
        summary_parts.append("\nRecent publications analysis:")
        for item in breakdown[:3]:  # Show top 3 keywords for brevity
            keyword = item.get("keyword", "unknown")
            relevance = item.get("relevance", "unknown")
            num_works = item.get("num_works", 0)
            contribution = item.get("total_contribution", 0)
            
            summary_parts.append(f"- '{keyword}' (relevance: {relevance}): {num_works} recent papers, score contribution: {contribution:.1f}")
            
            # Show top publications for this keyword
            top_pubs = item.get("top_publications", [])
            if top_pubs:
                venues = {}
                years = {}
                for pub in top_pubs[:3]:  # Top 3 publications
                    venue = pub.get("venue", "Unknown")
                    year = pub.get("year", "Unknown")
                    venues[venue] = venues.get(venue, 0) + 1
                    years[str(year)] = years.get(str(year), 0) + 1
                
                venue_summary = ", ".join([f"{v} ({c})" for v, c in sorted(venues.items(), key=lambda x: x[1], reverse=True)])
                year_summary = ", ".join([f"{y} ({c})" for y, c in sorted(years.items(), reverse=True)])
                summary_parts.append(f"  Top venues: {venue_summary}")
                summary_parts.append(f"  Publication years: {year_summary}")
    
    return "\n".join(summary_parts)


def compute_score_with_llm(sig: PaperSignals, title: str, abstract: str, authors: list[str], topic_details: dict = None) -> PaperScore:
    """
    Use LLM to provide structured inputs for FWCI-based scoring instead of arbitrary scores.
    
    Args:
        sig: PaperSignals containing all the metrics
        title: Paper title
        abstract: Paper abstract  
        authors: List of authors
        topic_details: Detailed breakdown of topic scoring
    
    Returns:
        PaperScore with evidence-based FWCI score and components
    """
    if not config.openai_api_key or OpenAI is None:
        # Fallback: simple heuristic scoring if LLM not available
        logger = logging.getLogger("arxiv_parser")
        logger.warning("OpenAI not configured; using fallback scoring")
        return _fallback_scoring(sig)
    
    if topic_details is None:
        topic_details = {}
    
    # Prepare information for LLM
    author_info = f"Average H-index: {sig.author_h_index_avg or 'Unknown'}, Recent works: {sig.author_recent_works_avg or 'Unknown'}"
    
    # Create curated topic summary
    topic_summary = _create_topic_summary(topic_details, sig.topic_activity_score)
    
    prompt = f"""You are a scientific impact assessment expert. Analyze this paper and provide structured quantitative inputs for Field-Weighted Citation Impact (FWCI) calculation.

DO NOT provide a final score. Instead, provide specific numerical assessments for each factor that will be used in an evidence-based citation prediction model.

Title: {title}

Abstract: {abstract}

Authors: {', '.join(authors[:5])}{'...' if len(authors) > 5 else ''}

Current Metrics:
- {author_info}
- Venue Impact: {sig.venue_two_year_mean_citedness or 'Unknown'} (2-year mean citedness)

Topic Activity Analysis:
{topic_summary}

Provide numerical assessments (0.0-1.0 scale unless specified) for these factors:

1. NOVELTY_SCORE: How methodologically/theoretically novel is this work? (0.0-1.0)
   - 0.0-0.2: Incremental work, standard methods
   - 0.3-0.5: Some new insights or applications
   - 0.6-0.8: Notable innovation or new approach
   - 0.9-1.0: Groundbreaking, paradigm-shifting work

2. AUTHOR_QUALITY_MULTIPLIER: Quality multiplier based on author track record (0.5-2.0)
   - 0.5-0.7: Early career or limited track record
   - 0.8-1.2: Established researchers, normal trajectory
   - 1.3-1.7: High-impact researchers with strong H-index
   - 1.8-2.0: Top-tier researchers, exceptional track record

3. FIELD_ACTIVITY_MULTIPLIER: How "hot" is this research area? (0.5-2.0)
   - 0.5-0.7: Declining or niche field
   - 0.8-1.2: Steady interest, normal field activity
   - 1.3-1.7: Growing field, increasing publications
   - 1.8-2.0: Extremely active, rapidly expanding field

4. VENUE_QUALITY_FACTOR: Venue prestige and visibility factor (0.5-2.0)
   - 0.5-0.7: Lower-tier venues, limited visibility
   - 0.8-1.2: Standard academic venues
   - 1.3-1.7: High-impact journals/conferences
   - 1.8-2.0: Top-tier, high-visibility venues

5. REPRODUCIBILITY_SCORE: Likelihood of follow-up work (0.0-1.0)
   - 0.0-0.2: Difficult to reproduce, limited follow-up potential
   - 0.3-0.5: Some reproducibility concerns
   - 0.6-0.8: Clear methods, good follow-up potential
   - 0.9-1.0: Excellent reproducibility, high follow-up likelihood

6. IMPACT_BREADTH: Potential to influence multiple subfields (0.0-1.0)
   - 0.0-0.2: Very specialized, narrow impact
   - 0.3-0.5: Moderate interdisciplinary relevance
   - 0.6-0.8: Good cross-field applicability
   - 0.9-1.0: Broad impact across multiple fields

Provide your assessment as:
REASONING: [Detailed quantitative reasoning for each factor]
NOVELTY_SCORE: [0.0-1.0]
AUTHOR_QUALITY_MULTIPLIER: [0.5-2.0]
FIELD_ACTIVITY_MULTIPLIER: [0.5-2.0]
VENUE_QUALITY_FACTOR: [0.5-2.0]
REPRODUCIBILITY_SCORE: [0.0-1.0]
IMPACT_BREADTH: [0.0-1.0]
"""
    
    try:
        client = OpenAI(api_key=config.openai_api_key)
        resp = client.chat.completions.create(
            model=config.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500  # Increased for comprehensive reasoning
        )
        
        response_text = resp.choices[0].message.content.strip()
        
        # Extract structured inputs from response
        structured_inputs = _extract_structured_inputs_from_llm_response(response_text)
        
        # Calculate FWCI-based score using the structured inputs
        fwci_score = _calculate_fwci_from_structured_inputs(structured_inputs, sig)
        
        # Create components for transparency
        components = {
            "fwci_score": fwci_score,
            "structured_inputs": structured_inputs,
            "author_h_index": sig.author_h_index_avg or 0.0,
            "author_recent": sig.author_recent_works_avg or 0.0,
            "topic_activity": sig.topic_activity_score,
            "llm_reasoning": response_text
        }
        
        return PaperScore(raw=fwci_score, components=components)
        
    except Exception as e:
        logger = logging.getLogger("arxiv_parser")
        logger.warning(f"LLM scoring failed: {e}, using fallback")
        return _fallback_scoring(sig)


def _extract_structured_inputs_from_llm_response(response_text: str) -> dict:
    """Extract structured numerical inputs from LLM response."""
    import re
    
    inputs = {}
    
    # Define the input patterns to extract
    patterns = {
        'novelty_score': r'NOVELTY_SCORE:\s*([0-9]+\.?[0-9]*)',
        'author_quality_multiplier': r'AUTHOR_QUALITY_MULTIPLIER:\s*([0-9]+\.?[0-9]*)',
        'field_activity_multiplier': r'FIELD_ACTIVITY_MULTIPLIER:\s*([0-9]+\.?[0-9]*)',
        'venue_quality_factor': r'VENUE_QUALITY_FACTOR:\s*([0-9]+\.?[0-9]*)',
        'reproducibility_score': r'REPRODUCIBILITY_SCORE:\s*([0-9]+\.?[0-9]*)',
        'impact_breadth': r'IMPACT_BREADTH:\s*([0-9]+\.?[0-9]*)'
    }
    
    # Extract each input with validation
    for key, pattern in patterns.items():
        match = re.search(pattern, response_text)
        if match:
            try:
                value = float(match.group(1))
                # Apply range validation
                if key in ['novelty_score', 'reproducibility_score', 'impact_breadth']:
                    inputs[key] = max(0.0, min(1.0, value))  # 0-1 range
                else:  # multipliers
                    inputs[key] = max(0.5, min(2.0, value))  # 0.5-2.0 range
            except ValueError:
                # Use default values if parsing fails
                if key in ['novelty_score', 'reproducibility_score', 'impact_breadth']:
                    inputs[key] = 0.5  # Neutral for 0-1 scale
                else:
                    inputs[key] = 1.0  # Neutral for multipliers
        else:
            # Use default values if not found
            if key in ['novelty_score', 'reproducibility_score', 'impact_breadth']:
                inputs[key] = 0.5  # Neutral for 0-1 scale
            else:
                inputs[key] = 1.0  # Neutral for multipliers
    
    return inputs


def _calculate_fwci_from_structured_inputs(inputs: dict, sig: PaperSignals) -> float:
    """Calculate FWCI score using structured inputs from LLM and empirical data."""
    try:
        from .citation_models import CitationPredictor, CitationPredictionInputs
        
        # Convert our data to citation model inputs
        citation_inputs = CitationPredictionInputs(
            author_h_index_max=sig.author_h_index_avg or 0.0,
            author_h_index_avg=sig.author_h_index_avg or 0.0,
            venue_impact_factor=sig.venue_two_year_mean_citedness or 1.0,
            topic_novelty_score=inputs.get('novelty_score', 0.5) * 10,  # Scale to 0-10
            field_activity_score=sig.topic_activity_score or 0.0,
            years_since_publication=0.1  # New paper
        )
        
        # Use ensemble model for base prediction
        base_prediction, model_components = CitationPredictor.ensemble_model(citation_inputs)
        
        # Apply LLM-derived multipliers
        author_multiplier = inputs.get('author_quality_multiplier', 1.0)
        field_multiplier = inputs.get('field_activity_multiplier', 1.0)
        venue_multiplier = inputs.get('venue_quality_factor', 1.0)
        
        # Additional factors
        reproducibility = inputs.get('reproducibility_score', 0.5)
        impact_breadth = inputs.get('impact_breadth', 0.5)
        
        # Combine multipliers - be conservative to avoid overestimation
        combined_multiplier = (
            author_multiplier * 0.3 +
            field_multiplier * 0.3 +
            venue_multiplier * 0.2 +
            (1 + reproducibility * 0.5) * 0.1 +
            (1 + impact_breadth * 0.5) * 0.1
        )
        
        # Convert predicted citations to FWCI equivalent
        # FWCI of 1.0 = expected citations for field
        # Assume baseline expected citations ~ 5-10 for typical papers
        expected_citations_baseline = 7.0
        fwci_score = (base_prediction * combined_multiplier) / expected_citations_baseline
        
        # Cap at reasonable FWCI range (0.1 to 5.0)
        return max(0.1, min(5.0, fwci_score))
        
    except ImportError:
        # Fallback calculation if citation_models not available
        logger = logging.getLogger("arxiv_parser")
        logger.warning("Citation models not available, using simplified calculation")
        
        # Simple calculation based on LLM inputs
        base_score = 1.0  # Neutral FWCI
        
        # Apply multipliers
        novelty_factor = 1 + inputs.get('novelty_score', 0.5)
        author_factor = inputs.get('author_quality_multiplier', 1.0)
        field_factor = inputs.get('field_activity_multiplier', 1.0)
        venue_factor = inputs.get('venue_quality_factor', 1.0)
        
        combined_score = base_score * novelty_factor * author_factor * field_factor * venue_factor * 0.5
        
        return max(0.1, min(5.0, combined_score))


def _extract_score_from_llm_response(response_text: str) -> float:
    """Extract numerical score from LLM response."""
    import re
    
    # Look for "PREDICTED_FWCI: X.X" pattern
    score_match = re.search(r'PREDICTED_FWCI:\s*([0-9]+\.?[0-9]*)', response_text)
    if score_match:
        try:
            score = float(score_match.group(1))
            return max(0.0, min(5.0, score))  # Clamp to valid FWCI range
        except ValueError:
            pass
    
    # Fallback: look for "SCORE: X.X" pattern (legacy)
    score_match = re.search(r'SCORE:\s*([0-9]+\.?[0-9]*)', response_text)
    if score_match:
        try:
            score = float(score_match.group(1))
            # Convert 0-10 scale to 0-5 FWCI scale if needed
            if score > 5.0:
                score = score / 2.0
            return max(0.0, min(5.0, score))
        except ValueError:
            pass
    
    # Fallback: look for any number in the response that could be a score
    numbers = re.findall(r'\b([0-9]+\.?[0-9]*)\b', response_text)
    for num_str in numbers:
        try:
            num = float(num_str)
            if 0.0 <= num <= 5.0:
                return num
            elif 0.0 <= num <= 10.0:
                return num / 2.0  # Convert 10-scale to FWCI-scale
        except ValueError:
            continue
    
    # Default fallback score (neutral FWCI)
    return 1.0


def _fallback_scoring(sig: PaperSignals) -> PaperScore:
    """Simple fallback scoring when LLM is not available."""
    # Basic heuristic: combine metrics with simple weights
    author_score = (sig.author_h_index_avg or 0) * 0.1 + (sig.author_recent_works_avg or 0) * 0.1
    topic_score = min(sig.topic_activity_score * 0.01, 3.0)  # Scale down topic score
    
    base_score = author_score + topic_score
    final_score = base_score
    
    components = {
        "fallback_score": final_score,
        "author_component": author_score,
        "topic_component": topic_score,
    }
    
    return PaperScore(raw=final_score, components=components)


# --- Topic activity scoring with LLM-generated keywords and log-normal projection ---

def extract_keywords_with_llm(title: str, abstract: str, num_keywords: int = 5) -> list[dict]:
    """Use LLM to extract keywords with relevance scores for topic search.
    
    Returns:
        List of dicts with 'keyword' and 'relevance' (0.0-1.0) keys
    """
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
                if clean_word not in [kw['keyword'] for kw in keywords]:
                    keywords.append({'keyword': clean_word, 'relevance': 0.7})  # Default relevance
                if len(keywords) >= num_keywords:
                    break
        return keywords[:num_keywords]

    client = OpenAI(api_key=config.openai_api_key)
    prompt = (
        f"Extract {num_keywords} most relevant scientific keywords for this paper that would be useful "
        f"for finding similar research articles. Focus on specific technical terms, materials, methods, "
        f"or phenomena.\n\n"
        f"For each keyword, also provide a relevance score from 0.0 to 1.0 indicating how central "
        f"the keyword is to the paper's main contribution:\n"
        f"- 1.0: Core concept, central to the paper's main findings\n"
        f"- 0.8: Important related concept, directly relevant\n"
        f"- 0.6: Supporting concept, somewhat relevant\n"
        f"- 0.4: Background concept, tangentially related\n"
        f"- 0.2: Broad field concept, very general relevance\n\n"
        f"Format each line as: keyword_phrase | relevance_score\n"
        f"Example: quantum entanglement | 0.9\n\n"
        f"Title: {title}\n\nAbstract: {abstract}"
    )
    
    try:
        resp = client.chat.completions.create(
            model=config.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        keywords_text = resp.choices[0].message.content.strip()
        
        keywords = []
        for line in keywords_text.split('\n'):
            if '|' in line:
                parts = line.split('|')
                if len(parts) >= 2:
                    keyword = parts[0].strip().lower()
                    try:
                        relevance = float(parts[1].strip())
                        relevance = max(0.0, min(1.0, relevance))  # Clamp to [0,1]
                        keywords.append({'keyword': keyword, 'relevance': relevance})
                    except ValueError:
                        keywords.append({'keyword': keyword, 'relevance': 0.5})  # Default if parsing fails
            elif line.strip():
                # Fallback: treat as keyword with default relevance
                keywords.append({'keyword': line.strip().lower(), 'relevance': 0.5})
        
        return keywords[:num_keywords]
    except Exception as e:
        logger = logging.getLogger("arxiv_parser")
        logger.warning("LLM keyword extraction failed: %s", e)
        return []


def log_normal_projected_citations(journal_impact_factor: float, years_since_publication: float) -> float:
    """
    DEPRECATED: Simple log-normal projection (kept for backwards compatibility).
    Use get_improved_citation_prediction() from citation_models.py for better accuracy.
    
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
    Calculate projected citations for a work using configurable prediction methods.
    
    Uses the global citation prediction configuration to determine which method to use.
    Supports legacy log-normal, improved evidence-based models, or auto-selection.
    
    Args:
        work: OpenAlex work dict containing host_venue and publication info
    
    Returns:
        Projected citations count
    """
    try:
        from .citation_config import get_citation_prediction
        return get_citation_prediction(work)
    except ImportError:
        # Fallback to legacy method if citation_config not available
        logger = logging.getLogger("arxiv_parser")
        logger.warning("Citation config not available, using legacy method")
        return _legacy_projected_citations_for_work(work)
    except Exception as e:
        logger = logging.getLogger("arxiv_parser")
        logger.warning(f"Citation prediction failed: {e}, using legacy method")
        return _legacy_projected_citations_for_work(work)


def _legacy_projected_citations_for_work(work: dict) -> float:
    """
    Legacy citation prediction method (kept for backwards compatibility).
    
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


def compute_topic_activity_score_with_keywords_detailed(keywords: list[dict], max_works_per_keyword: int = 50) -> tuple[float, list]:
    """
    Compute topic activity score using LLM-generated keywords with relevance scores and return detailed breakdown.
    
    For each keyword:
    1. Search for recent articles containing the keyword
    2. For each article found, calculate: projected_citations * time_decay * keyword_relevance
    3. Sum all contributions and track details
    
    Args:
        keywords: List of keyword dicts with 'keyword' and 'relevance' keys
        max_works_per_keyword: Maximum number of works to consider per keyword
    
    Returns:
        Tuple of (total score, list of keyword details)
    """
    from . import openalex_client
    
    total_score = 0.0
    keyword_details = []
    logger = logging.getLogger("arxiv_parser")
    
    # Search from 3 years ago to capture recent activity
    try:
        UTC = dt.UTC
    except AttributeError:
        from datetime import timezone as _tz
        UTC = _tz.utc
    
    from_date = (dt.datetime.now(UTC) - dt.timedelta(days=3*365)).strftime("%Y-%m-%d")
    
    for kw_data in keywords:
        keyword = kw_data.get('keyword', '') if isinstance(kw_data, dict) else str(kw_data)
        relevance = kw_data.get('relevance', 1.0) if isinstance(kw_data, dict) else 1.0
        
        try:
            # Search for works containing this keyword
            works = search_works_by_keyword(keyword, from_date, max_works_per_keyword)
            logger.debug(f"Found {len(works)} works for keyword '{keyword}' (relevance: {relevance:.2f})")
            
            keyword_score = 0.0
            publications_info = []
            
            for work in works:
                projected_cites = projected_citations_for_work(work)
                pub_datetime = publication_datetime_from_work(work)
                decay = time_decay(pub_datetime)
                # Apply relevance multiplier to scale the contribution
                contribution = projected_cites * decay * relevance
                
                # Capture publication details
                venue_name = work.get("host_venue", {}).get("display_name", "Unknown venue")
                pub_year = pub_datetime.year
                publications_info.append({
                    "venue": venue_name,
                    "year": pub_year,
                    "projected_citations": projected_cites,
                    "time_decay": decay,
                    "relevance_multiplier": relevance,
                    "contribution": contribution
                })
                
                keyword_score += contribution
                total_score += contribution
            
            keyword_details.append({
                "keyword": keyword,
                "relevance": relevance,
                "num_works": len(works),
                "total_contribution": keyword_score,
                "top_publications": publications_info[:5]  # Keep top 5 for brevity
            })
                
        except Exception as e:
            logger.warning(f"Error processing keyword '{keyword}': {e}")
            keyword_details.append({
                "keyword": keyword,
                "relevance": relevance,
                "num_works": 0,
                "total_contribution": 0.0,
                "error": str(e)
            })
            continue
    
    return total_score, keyword_details


def compute_topic_activity_score_with_keywords(keywords: list[dict], max_works_per_keyword: int = 50) -> float:
    """
    Compute topic activity score using LLM-generated keywords with relevance scores.
    
    For each keyword:
    1. Search for recent articles containing the keyword
    2. For each article found, calculate: projected_citations * time_decay * keyword_relevance
    3. Sum all contributions
    
    Args:
        keywords: List of keyword dicts with 'keyword' and 'relevance' keys
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
    
    for kw_data in keywords:
        keyword = kw_data.get('keyword', '') if isinstance(kw_data, dict) else str(kw_data)
        relevance = kw_data.get('relevance', 1.0) if isinstance(kw_data, dict) else 1.0
        
        try:
            # Search for works containing this keyword
            works = search_works_by_keyword(keyword, from_date, max_works_per_keyword)
            logger.debug(f"Found {len(works)} works for keyword '{keyword}' (relevance: {relevance:.2f})")
            
            for work in works:
                projected_cites = projected_citations_for_work(work)
                pub_datetime = publication_datetime_from_work(work)
                decay = time_decay(pub_datetime)
                # Apply relevance multiplier to scale the contribution
                contribution = projected_cites * decay * relevance
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


def explain_score(sig: PaperSignals, score_result: PaperScore, topic_details: dict = None) -> dict:
    """Return explanation of the improved LLM-based score calculation for templating.

    Structure:
    {
      'signals': {... raw values ...},
      'components': {... score components ...},
      'final_score': float,
      'llm_reasoning': str (if available),
      'structured_inputs': dict (if available),
      'topic_activity_breakdown': dict (if available)
    }
    """
    
    explanation = {
        "signals": {
            "author_h_index_avg": sig.author_h_index_avg,
            "author_recent_works_avg": sig.author_recent_works_avg,
            "venue_two_year_mean_citedness": sig.venue_two_year_mean_citedness,
            "published": sig.published.isoformat(),
            "topic_activity_score": sig.topic_activity_score,
            "scoring_method": "llm_structured_fwci_based",
        },
        "components": score_result.components,
        "final_score": score_result.raw,
    }
    
    # Include LLM reasoning if available
    if "llm_reasoning" in score_result.components:
        explanation["llm_reasoning"] = score_result.components["llm_reasoning"]
    
    # Include structured inputs if available
    if "structured_inputs" in score_result.components:
        explanation["structured_inputs"] = score_result.components["structured_inputs"]
    
    # Include topic activity breakdown if available
    if topic_details:
        explanation["topic_activity_breakdown"] = topic_details
    
    return explanation


def compute_score(sig: PaperSignals, title: str = "", abstract: str = "", authors: list[str] = None, topic_details: dict = None) -> PaperScore:
    """
    Compatibility wrapper for the new LLM-based scoring system.
    
    Args:
        sig: PaperSignals containing metrics
        title: Paper title (optional for backwards compatibility)
        abstract: Paper abstract (optional for backwards compatibility)  
        authors: List of authors (optional for backwards compatibility)
        topic_details: Topic calculation details (optional)
    
    Returns:
        PaperScore with LLM-determined score
    """
    if authors is None:
        authors = []
    if topic_details is None:
        topic_details = {}
    
    return compute_score_with_llm(sig, title, abstract, authors, topic_details)


def compute_paper_topic_activity_score(title: str, abstract: str, num_keywords: int = 5, max_works_per_keyword: int = 50) -> tuple[float, dict]:
    """
    Main function to compute topic activity score for a paper using LLM-generated keywords with relevance scores.
    
    Args:
        title: Paper title
        abstract: Paper abstract
        num_keywords: Number of keywords to extract with LLM
        max_works_per_keyword: Maximum works to analyze per keyword
    
    Returns:
        Tuple of (topic activity score, details dict with keywords and breakdown)
    """
    # Step 1: Extract keywords with relevance scores using LLM
    keywords_with_relevance = extract_keywords_with_llm(title, abstract, num_keywords)
    
    if not keywords_with_relevance:
        logging.getLogger("arxiv_parser").warning("No keywords extracted, returning zero topic score")
        return 0.0, {"keywords": [], "breakdown": []}
    
    # Step 2: Compute activity score using these keywords and capture details
    score, details = compute_topic_activity_score_with_keywords_detailed(keywords_with_relevance, max_works_per_keyword)
    
    # Extract just keyword names for backward compatibility in the summary
    keyword_names = [kw.get('keyword', '') if isinstance(kw, dict) else str(kw) for kw in keywords_with_relevance]
    
    return score, {
        "keywords": keyword_names,
        "keywords_with_relevance": keywords_with_relevance,  # Include relevance data
        "breakdown": details,
        "total_score": score
    }
