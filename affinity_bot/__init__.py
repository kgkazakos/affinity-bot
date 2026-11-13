"""affinity-bot · CPR Orbital #36 (Season 6)

LLM-powered affinity diagramming and theme clustering from raw research
notes — observations, interview snippets, quotes, sticky-note dumps.

Design contract
---------------
* The rule-based core (TF-IDF vectorisation + KMeans clustering + top-term
  labelling) runs fully offline with zero API calls and produces a useful
  first-pass affinity diagram on its own.
* The LLM enrichment layer (``llm.py``) is **on by default** across three
  backends — Anthropic Claude Opus, Google Gemini 4 Argon, OpenAI GPT-5.5 —
  via DSPy/LiteLLM. It generates human-readable theme labels and short
  rationale strings. With no API key (or ``--no-llm``), the rule-based
  result is returned unchanged.
* Honest framing: the output is a starting point for a researcher's
  affinity wall, not a finished thematic analysis. Outliers are flagged,
  not silently absorbed. Theme labels are heuristics, not validated codes.

Public API
----------
``cluster_notes(notes, config) -> AffinityResult``
    The end-to-end entry point: extract → cluster → label → optionally enrich.

CLI: ``affinity-bot cluster <input>`` / ``backends`` / ``example``.
UI:  ``streamlit run affinity_bot/app.py``.
"""

__version__ = "0.1.0"

from affinity_bot.models import (
    AffinityConfig,
    AffinityResult,
    Cluster,
    Note,
)
from affinity_bot.pipeline import cluster_notes

__all__ = [
    "AffinityConfig",
    "AffinityResult",
    "Cluster",
    "Note",
    "cluster_notes",
    "__version__",
]
