"""Optional LLM enrichment for affinity-bot.

The rule-based core produces working clusters and terse keyword labels.
This layer is **on by default** and rewrites each cluster's label and
rationale into something a researcher would actually write in an
affinity diagram — a short noun phrase, and one sentence of context.

It is strictly additive and never destructive. If DSPy / LiteLLM is not
installed, or no API key is present, or the LLM call fails, the
rule-based cluster is returned unchanged and a warning is appended to
``result.warnings``. The tool never hard-fails on a missing key.
``--no-llm`` forces the offline path explicitly.

Backends (LLM is ON BY DEFAULT across all three):

    anthropic  ->  anthropic/claude-opus-4-7      (Claude Opus)
    gemini     ->  gemini/gemini-4-argon  (Gemini 4 Argon)
    openai     ->  openai/gpt-5.5                  (GPT-5.5)

Model strings are LiteLLM-style ``provider/model`` identifiers — edit
the ``BACKENDS`` map below if your account exposes different names.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from affinity_bot.models import AffinityConfig, Cluster, Note

logger = logging.getLogger(__name__)

BACKENDS: dict[str, str] = {
    "anthropic": "anthropic/claude-opus-4-7",
    "gemini": "gemini/gemini-4-argon",
    "openai": "openai/gpt-5.5",
}

# Priority order when ``llm_backend`` is not pinned: try anthropic first
# (highest quality on this task in practice), then gemini, then openai.
_BACKEND_PRIORITY: tuple[str, ...] = ("anthropic", "gemini", "openai")

_API_KEY_ENV: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
}


@dataclass
class EnrichmentOutcome:
    """Result of trying to enrich a clustering with LLM labels."""

    enriched_clusters: list[Cluster]
    used: bool
    backend: Optional[str]
    warnings: list[str]


def enrich_clusters(
    notes: list[Note],
    clusters: list[Cluster],
    config: AffinityConfig,
) -> EnrichmentOutcome:
    """Try to rewrite cluster labels and rationale via an LLM backend.

    Always returns a valid outcome — never raises. If anything fails, the
    rule-based clusters are passed through unchanged and the reason is
    recorded in ``warnings``.
    """
    if not config.use_llm:
        return EnrichmentOutcome(clusters, False, None, [])

    backend = _resolve_backend(config.llm_backend)
    if backend is None:
        return EnrichmentOutcome(
            clusters,
            False,
            None,
            [
                "LLM enrichment skipped: no API key found for anthropic, "
                "gemini, or openai. Falling back to rule-based labels."
            ],
        )

    model = config.llm_model or BACKENDS[backend]

    try:
        import dspy  # type: ignore
    except ImportError:
        return EnrichmentOutcome(
            clusters,
            False,
            None,
            [
                "LLM enrichment skipped: dspy / litellm not installed. "
                "Install with `pip install affinity-bot` (LLM deps are "
                "core, not extras)."
            ],
        )

    try:
        lm = dspy.LM(model=model, temperature=0.2, max_tokens=300)
        dspy.configure(lm=lm)
    except Exception as exc:  # noqa: BLE001
        return EnrichmentOutcome(
            clusters,
            False,
            None,
            [f"LLM enrichment skipped: DSPy could not configure {model}: {exc}"],
        )

    notes_by_id = {n.id: n for n in notes}
    enriched: list[Cluster] = []
    warnings: list[str] = []
    for cluster in clusters:
        if cluster.is_outlier or not cluster.note_ids:
            enriched.append(cluster)
            continue
        try:
            new_label, new_rationale = _label_one_cluster(
                cluster, notes_by_id
            )
            enriched.append(
                Cluster(
                    cluster_id=cluster.cluster_id,
                    label=new_label,
                    rationale=new_rationale,
                    note_ids=list(cluster.note_ids),
                    keyword_signature=list(cluster.keyword_signature),
                    label_source="llm",
                    is_outlier=False,
                )
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"LLM enrichment failed for cluster #{cluster.cluster_id}: "
                f"{exc}. Kept rule-based label."
            )
            enriched.append(cluster)

    return EnrichmentOutcome(enriched, True, backend, warnings)


def _label_one_cluster(
    cluster: Cluster, notes_by_id: dict[str, Note]
) -> tuple[str, str]:
    """Run a single DSPy ChainOfThought call to label one cluster."""
    import dspy  # type: ignore

    class AffinityLabel(dspy.Signature):  # type: ignore
        """Generate a researcher-style theme label and a one-sentence rationale
        from a set of related research notes.

        Rules:
        - Label: a short noun phrase (3-7 words). No leading articles.
          Title case. No trailing punctuation.
        - Rationale: a single sentence (~20-30 words) in plain language,
          describing what unifies these notes. Do not enumerate the notes.
        - Be specific and grounded — never use generic terms like
          "various themes" or "miscellaneous observations".
        """

        notes_block: str = dspy.InputField(
            desc="Numbered research notes belonging to one cluster."
        )
        keyword_signature: str = dspy.InputField(
            desc="Top distinguishing keywords from TF-IDF as supporting evidence."
        )
        label: str = dspy.OutputField(
            desc="Short noun-phrase theme label (3-7 words, title case)."
        )
        rationale: str = dspy.OutputField(
            desc="One-sentence plain-language rationale for the cluster."
        )

    # Build the prompt body. Cap the number of sample notes for token safety;
    # 12 is more than enough signal for any sensible cluster.
    sample_ids = cluster.note_ids[:12]
    lines = []
    for i, nid in enumerate(sample_ids, start=1):
        note = notes_by_id.get(nid)
        if note is None:
            continue
        # Trim absurdly long notes to keep prompt bounded
        text = note.text if len(note.text) <= 360 else note.text[:357] + "..."
        lines.append(f"{i}. {text}")
    notes_block = "\n".join(lines) if lines else "(no notes)"
    keyword_signature = (
        ", ".join(cluster.keyword_signature[:8])
        if cluster.keyword_signature
        else "(no keywords)"
    )

    program = dspy.ChainOfThought(AffinityLabel)
    pred = program(notes_block=notes_block, keyword_signature=keyword_signature)

    label = (getattr(pred, "label", "") or "").strip()
    rationale = (getattr(pred, "rationale", "") or "").strip()

    label = _sanitise_label(label) or cluster.label
    rationale = _sanitise_rationale(rationale) or cluster.rationale
    return label, rationale


def _sanitise_label(label: str) -> str:
    if not label:
        return ""
    label = label.strip().strip("\"'`")
    # Drop trailing sentence punctuation
    while label and label[-1] in {".", ",", ";", ":", "—", "-"}:
        label = label[:-1].rstrip()
    # Hard cap on length
    return label[:80]


def _sanitise_rationale(rationale: str) -> str:
    if not rationale:
        return ""
    rationale = rationale.strip().strip("\"'`")
    # Collapse any newlines into spaces
    rationale = " ".join(rationale.split())
    # Hard cap
    if len(rationale) > 280:
        rationale = rationale[:277].rstrip() + "..."
    if not rationale.endswith((".", "!", "?")):
        rationale = rationale + "."
    return rationale


def _resolve_backend(preferred: Optional[str]) -> Optional[str]:
    """Pick a backend whose API key is set in the environment."""
    if preferred:
        envs = _API_KEY_ENV.get(preferred, ())
        if any(os.environ.get(k) for k in envs):
            return preferred
        return None
    for backend in _BACKEND_PRIORITY:
        envs = _API_KEY_ENV.get(backend, ())
        if any(os.environ.get(k) for k in envs):
            return backend
    return None


def available_backends() -> dict[str, bool]:
    """Return a {backend: configured} map — useful for CLI ``backends`` output."""
    out: dict[str, bool] = {}
    for backend in _BACKEND_PRIORITY:
        envs = _API_KEY_ENV.get(backend, ())
        out[backend] = any(os.environ.get(k) for k in envs)
    return out
