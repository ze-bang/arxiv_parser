from __future__ import annotations

from typing import Optional

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None  # type: ignore

from .config import config


SYSTEM_PROMPT = (
    "You are a physics research assistant specializing in condensed matter theory. Summarize the paper for fellow condensed matter physicists. "
    "Highlight: (1) key physics concepts and theoretical/experimental methods, (2) main results and their significance, "
    "(3) novelty compared to existing work, (4) potential impact on the field, and (5) connections to current research trends. "
    "Be specific about the physical systems, phenomena, or materials studied. Contextualize the work within the broader field."
)


def summarize_with_llm(title: str, abstract: str, score_components: dict[str, float], model: Optional[str] = None) -> str:
    if not config.openai_api_key or OpenAI is None:
        # Fallback summary
        return f"{title}\n\nSummary: {abstract[:800]}...\n(LLM not configured; showing abstract snippet.)"

    m = model or config.openai_model
    client = OpenAI(api_key=config.openai_api_key)
    
    # Filter numeric components for display and include key metrics
    numeric_components = {}
    for k, v in score_components.items():
        if isinstance(v, (int, float)) and k != 'llm_reasoning':
            numeric_components[k] = v
    
    components_str = ", ".join(f"{k}: {v:.2f}" for k, v in numeric_components.items())
    
    # Include LLM reasoning if available for context
    llm_reasoning = score_components.get('llm_reasoning', '')
    reasoning_context = ""
    if llm_reasoning:
        # Extract key insights from the reasoning for context
        reasoning_lines = llm_reasoning.split('\n')[:5]  # First few lines
        reasoning_context = f"\n\nImpact Assessment Context: {' '.join(reasoning_lines)}"
    
    prompt = (
        f"Title: {title}\n\nAbstract: {abstract}\n\n"
        f"Quantitative Impact Metrics: {components_str}.{reasoning_context}\n\n"
        "Provide a comprehensive summary that includes:\n"
        "1. Core physics and methods\n"
        "2. Key results and physical insights\n" 
        "3. Significance and novelty in the field\n"
        "4. Potential research impact and follow-up directions\n"
        "Write 5-7 informative sentences with specific physics terminology."
    )
    try:
        resp = client.chat.completions.create(
            model=m,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"{title}\n\nSummary unavailable due to LLM error: {e}. Abstract: {abstract[:800]}..."
