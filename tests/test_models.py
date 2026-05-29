"""Tests for affinity_bot.models."""

from __future__ import annotations

import pytest

from affinity_bot.models import (
    AffinityConfig,
    AffinityResult,
    Cluster,
    Note,
)


def test_note_to_dict_minimal():
    n = Note(id="n0001", text="hello world")
    assert n.to_dict() == {"id": "n0001", "text": "hello world"}


def test_note_to_dict_with_source():
    n = Note(id="n0002", text="hi", source="P03")
    out = n.to_dict()
    assert out["source"] == "P03"
    assert out["id"] == "n0002"


def test_cluster_size_and_dict():
    c = Cluster(
        cluster_id=1,
        label="onboarding · friction · steps",
        rationale="Shared terms: onboarding, friction, steps.",
        note_ids=["n0001", "n0002", "n0003"],
        keyword_signature=["onboarding", "friction", "steps"],
        label_source="rule",
    )
    assert c.size == 3
    out = c.to_dict()
    assert out["size"] == 3
    assert out["label_source"] == "rule"
    assert out["is_outlier"] is False


def test_cluster_outlier_flag():
    c = Cluster(
        cluster_id=-1,
        label="Outliers",
        rationale="—",
        note_ids=["n0042"],
        is_outlier=True,
    )
    assert c.is_outlier
    assert c.size == 1


def test_affinity_config_defaults():
    cfg = AffinityConfig()
    assert cfg.use_llm is True
    assert cfg.llm_backend is None
    assert cfg.min_clusters == 3
    assert cfg.max_clusters == 8
    assert cfg.outlier_distance_threshold == 0.85


def test_affinity_config_validates_bounds():
    with pytest.raises(ValueError):
        AffinityConfig(min_clusters=1)
    with pytest.raises(ValueError):
        AffinityConfig(min_clusters=6, max_clusters=4)
    with pytest.raises(ValueError):
        AffinityConfig(n_clusters=1)
    with pytest.raises(ValueError):
        AffinityConfig(outlier_distance_threshold=0.0)
    with pytest.raises(ValueError):
        AffinityConfig(outlier_distance_threshold=1.5)
    with pytest.raises(ValueError):
        AffinityConfig(keywords_per_cluster=0)


def test_affinity_config_backend_validation():
    AffinityConfig(llm_backend="anthropic")
    AffinityConfig(llm_backend="gemini")
    AffinityConfig(llm_backend="openai")
    with pytest.raises(ValueError):
        AffinityConfig(llm_backend="bogus")


def test_affinity_result_counts_and_cluster_for_note():
    notes = [Note(id=f"n000{i}", text=f"text {i}") for i in range(1, 4)]
    clusters = [
        Cluster(
            cluster_id=0,
            label="A",
            rationale="r",
            note_ids=["n0001", "n0002"],
            label_source="rule",
        ),
        Cluster(
            cluster_id=-1,
            label="Outliers",
            rationale="r",
            note_ids=["n0003"],
            is_outlier=True,
        ),
    ]
    res = AffinityResult(
        notes=notes,
        clusters=clusters,
        config=AffinityConfig(),
        notes_clustered=3,
        outlier_count=1,
    )
    assert res.cluster_count == 1
    assert res.cluster_for_note("n0001").label == "A"
    assert res.cluster_for_note("n0003").is_outlier
    assert res.cluster_for_note("n9999") is None


def test_affinity_result_to_dict_structure():
    res = AffinityResult(
        notes=[Note(id="n0001", text="hello")],
        clusters=[Cluster(0, "A", "r", ["n0001"])],
        config=AffinityConfig(),
        notes_clustered=1,
    )
    out = res.to_dict()
    assert "notes" in out and "clusters" in out and "config" in out
    assert out["notes_clustered"] == 1
    assert out["cluster_count"] == 1


def test_note_is_frozen():
    from dataclasses import FrozenInstanceError
    n = Note(id="n0001", text="x")
    with pytest.raises(FrozenInstanceError):
        n.text = "y"  # type: ignore[misc]
