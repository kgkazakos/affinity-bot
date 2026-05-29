"""End-to-end entry point: notes in, AffinityResult out."""

from __future__ import annotations

from collections.abc import Iterable

from affinity_bot.clusterer import cluster as _cluster_step
from affinity_bot.extract import notes_from_strings
from affinity_bot.labeler import build_clusters
from affinity_bot.llm import enrich_clusters
from affinity_bot.models import AffinityConfig, AffinityResult, Note


def cluster_notes(
    notes: Iterable[str] | list[Note],
    config: AffinityConfig | None = None,
) -> AffinityResult:
    """Cluster research notes into themes.

    Parameters
    ----------
    notes
        Either a list of raw strings (one per note) or a list of
        :class:`Note` objects with stable ids.
    config
        :class:`AffinityConfig`. If omitted, default settings are used
        (LLM on by default; graceful offline fallback).
    """
    cfg = config or AffinityConfig()

    # Normalise to list[Note]
    note_objs: list[Note]
    if not notes:
        note_objs = []
    elif isinstance(notes, list) and notes and isinstance(notes[0], Note):
        note_objs = list(notes)  # type: ignore[arg-type]
    else:
        note_objs = notes_from_strings(notes)  # type: ignore[arg-type]

    if not note_objs:
        return AffinityResult(
            notes=[],
            clusters=[],
            config=cfg,
            llm_used=False,
            llm_backend=None,
            notes_clustered=0,
            outlier_count=0,
            warnings=["No notes provided."],
        )

    if len(note_objs) < cfg.min_clusters * 2:
        # Too few notes for sensible clustering — return a single cluster of
        # all notes with a clear warning rather than failing.
        return _single_bucket_result(note_objs, cfg)

    assignment = _cluster_step(note_objs, cfg)
    clusters = build_clusters(note_objs, assignment, cfg)
    outcome = enrich_clusters(note_objs, clusters, cfg)

    outlier_count = sum(c.size for c in outcome.enriched_clusters if c.is_outlier)
    return AffinityResult(
        notes=note_objs,
        clusters=outcome.enriched_clusters,
        config=cfg,
        llm_used=outcome.used,
        llm_backend=outcome.backend,
        notes_clustered=len(note_objs),
        outlier_count=outlier_count,
        warnings=outcome.warnings,
    )


def _single_bucket_result(
    notes: list[Note], config: AffinityConfig
) -> AffinityResult:
    """Edge case: corpus is too small to cluster meaningfully."""
    from affinity_bot.models import Cluster  # local import to avoid cycle

    cluster = Cluster(
        cluster_id=0,
        label="All notes (corpus too small to cluster)",
        rationale=(
            f"Only {len(notes)} note(s) provided — affinity-bot needs at "
            f"least {config.min_clusters * 2} for sensible clustering. "
            f"All notes are grouped together; no automated theming was attempted."
        ),
        note_ids=[n.id for n in notes],
        keyword_signature=[],
        label_source="rule",
        is_outlier=False,
    )
    return AffinityResult(
        notes=notes,
        clusters=[cluster],
        config=config,
        llm_used=False,
        llm_backend=None,
        notes_clustered=len(notes),
        outlier_count=0,
        warnings=[
            f"Corpus too small: only {len(notes)} note(s); skipped clustering. "
            f"Add more notes (>= {config.min_clusters * 2}) for affinity grouping."
        ],
    )
