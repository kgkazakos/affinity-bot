"""Tests for affinity_bot.labeler."""

from __future__ import annotations

import numpy as np

from affinity_bot.clusterer import ClusterAssignment
from affinity_bot.labeler import build_clusters
from affinity_bot.models import AffinityConfig, Note


def _fake_assignment(note_ids, labels, vocab, centroids) -> ClusterAssignment:
    return ClusterAssignment(
        note_ids=list(note_ids),
        labels=list(labels),
        centroids=np.array(centroids),
        vocabulary=list(vocab),
        chosen_k=len({lab for lab in labels if lab >= 0}),
    )


def test_build_clusters_groups_by_label():
    notes = [Note(id=f"n000{i}", text=f"t{i}") for i in range(1, 5)]
    assignment = _fake_assignment(
        note_ids=[n.id for n in notes],
        labels=[0, 0, 1, 1],
        vocab=["alpha", "beta", "gamma"],
        centroids=[[0.9, 0.1, 0.0], [0.0, 0.2, 0.95]],
    )
    cfg = AffinityConfig(keywords_per_cluster=2, min_clusters=2, max_clusters=2)
    clusters = build_clusters(notes, assignment, cfg)
    # 2 real clusters, no outliers
    assert len(clusters) == 2
    by_id = {c.cluster_id: c for c in clusters}
    assert by_id[0].size == 2
    assert by_id[1].size == 2
    assert all(c.label_source == "rule" for c in clusters)


def test_build_clusters_label_uses_top_terms():
    notes = [Note(id="n0001", text="t"), Note(id="n0002", text="t")]
    assignment = _fake_assignment(
        note_ids=["n0001", "n0002"],
        labels=[0, 0],
        vocab=["onboarding", "friction", "noise"],
        centroids=[[0.9, 0.6, 0.05]],
    )
    cfg = AffinityConfig(keywords_per_cluster=2, min_clusters=2, max_clusters=2)
    clusters = build_clusters(notes, assignment, cfg)
    assert "onboarding" in clusters[0].label
    assert "friction" in clusters[0].label
    assert "·" in clusters[0].label


def test_build_clusters_outlier_bucket():
    notes = [Note(id="n0001", text="x"), Note(id="n0002", text="y")]
    assignment = _fake_assignment(
        note_ids=["n0001", "n0002"],
        labels=[0, -1],
        vocab=["foo", "bar"],
        centroids=[[0.5, 0.5]],
    )
    cfg = AffinityConfig(min_clusters=2, max_clusters=2)
    clusters = build_clusters(notes, assignment, cfg)
    outlier = next(c for c in clusters if c.is_outlier)
    assert outlier.cluster_id == -1
    assert outlier.label == "Outliers"
    assert outlier.note_ids == ["n0002"]
    assert "review individually" in outlier.rationale.lower()


def test_build_clusters_sorts_real_by_size_desc():
    notes = [Note(id=f"n000{i}", text="t") for i in range(1, 6)]
    assignment = _fake_assignment(
        note_ids=[n.id for n in notes],
        labels=[0, 1, 1, 1, 2],
        vocab=["a", "b"],
        centroids=[[1, 0], [0.5, 0.5], [0, 1]],
    )
    clusters = build_clusters(
        notes, assignment, AffinityConfig(min_clusters=2, max_clusters=3)
    )
    sizes = [c.size for c in clusters if not c.is_outlier]
    assert sizes == sorted(sizes, reverse=True)


def test_build_clusters_outliers_last():
    notes = [Note(id=f"n000{i}", text="t") for i in range(1, 4)]
    assignment = _fake_assignment(
        note_ids=[n.id for n in notes],
        labels=[0, -1, 0],
        vocab=["a"],
        centroids=[[1.0]],
    )
    clusters = build_clusters(notes, assignment, AffinityConfig(min_clusters=2, max_clusters=2))
    assert clusters[-1].is_outlier


def test_keyword_signature_respects_count():
    notes = [Note(id="n0001", text="t")]
    assignment = _fake_assignment(
        note_ids=["n0001"],
        labels=[0],
        vocab=["a", "b", "c", "d", "e"],
        centroids=[[1, 0.9, 0.8, 0.7, 0.6]],
    )
    cfg = AffinityConfig(keywords_per_cluster=3, min_clusters=2, max_clusters=2)
    clusters = build_clusters(notes, assignment, cfg)
    # keyword_signature stores at least max(3, keywords_per_cluster) terms internally
    assert len(clusters[0].keyword_signature) >= 3
    # But the displayed label uses only the configured count
    label = clusters[0].label
    assert "d" not in label and "e" not in label
