from __future__ import annotations

from typing import Optional

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None  # type: ignore

from .config import config


SYSTEM_PROMPT = (
    "You are a physics research assistant. Summarize the paper for a condensed matter physicist. "
    "Highlight novelty, methods, and potential impact, and briefly justify the impact score based on authors and topic prevalence."
)


def summarize_with_llm(title: str, abstract: str, score_components: dict[str, float], model: Optional[str] = None) -> str:
    if not config.openai_api_key or OpenAI is None:
        # Fallback summary
        return f"{title}\n\nSummary: {abstract[:800]}...\n(LLM not configured; showing abstract snippet.)"

    m = model or config.openai_model
    client = OpenAI(api_key=config.openai_api_key)
    components_str = ", ".join(f"{k}: {v:.2f}" for k, v in score_components.items())
    prompt = (
        f"Title: {title}\n\nAbstract: {abstract}\n\nScore components: {components_str}.\n"
        "Write 4-6 informative sentences."
    )
    try:
        resp = client.chat.completions.create(
            model=m,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"{title}\n\nSummary unavailable due to LLM error: {e}. Abstract: {abstract[:800]}..."
