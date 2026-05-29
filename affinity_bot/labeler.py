"""Rule-based label and rationale generation for clusters.

The labels produced here are intentionally terse and *honest about their
origin* — they're the top-weighted TF-IDF terms, joined with a separator.
The LLM layer rewrites them into proper noun phrases when enabled; this
module is the fallback that runs when no API key is present.
"""

from __future__ import annotations

from affinity_bot.clusterer import ClusterAssignment, top_terms_for_cluster
from affinity_bot.models import AffinityConfig, Cluster, Note

_OUTLIER_LABEL = "Outliers"
_OUTLIER_RATIONALE = (
    "Notes that did not fit any cluster cleanly — review individually."
)


def build_clusters(
    notes: list[Note],
    assignment: ClusterAssignment,
    config: AffinityConfig,
) -> list[Cluster]:
    """Group notes by cluster id and assign rule-based labels."""
    by_id: dict[int, list[str]] = {}
    for note_id, lab in zip(assignment.note_ids, assignment.labels, strict=False):
        by_id.setdefault(lab, []).append(note_id)

    clusters: list[Cluster] = []
    for cluster_id in sorted(by_id.keys()):
        note_ids = by_id[cluster_id]
        if cluster_id == -1:
            clusters.append(
                Cluster(
                    cluster_id=-1,
                    label=_OUTLIER_LABEL,
                    rationale=_OUTLIER_RATIONALE,
                    note_ids=note_ids,
                    keyword_signature=[],
                    label_source="rule",
                    is_outlier=True,
                )
            )
            continue
        terms = top_terms_for_cluster(
            cluster_id, assignment, top_k=max(3, config.keywords_per_cluster)
        )
        label = _format_rule_label(terms[: config.keywords_per_cluster])
        rationale = _format_rule_rationale(terms[: config.keywords_per_cluster])
        clusters.append(
            Cluster(
                cluster_id=cluster_id,
                label=label,
                rationale=rationale,
                note_ids=note_ids,
                keyword_signature=terms,
                label_source="rule",
                is_outlier=False,
            )
        )

    # Sort: real clusters first by size desc, outliers last
    clusters.sort(key=lambda c: (c.is_outlier, -c.size, c.cluster_id))
    return clusters


def _format_rule_label(terms: list[str]) -> str:
    if not terms:
        return "Unlabelled cluster"
    return " · ".join(terms)


def _format_rule_rationale(terms: list[str]) -> str:
    if not terms:
        return "Notes grouped by overall similarity."
    return "Shared terms: " + ", ".join(terms) + "."
