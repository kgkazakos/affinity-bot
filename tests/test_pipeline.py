"""End-to-end tests for affinity_bot.pipeline.cluster_notes."""

from __future__ import annotations

from affinity_bot.models import AffinityConfig, Note
from affinity_bot.pipeline import cluster_notes


def _three_theme_corpus() -> list[str]:
    return [
        "Onboarding flow felt overwhelming with too many steps",
        "I gave up creating an account because there were three forms",
        "Sign up took ages and I almost closed the tab",
        "The welcome screen explains nothing about the product",
        "Dashboard cards are pleasantly sized and calm",
        "The dashboard is the best part of the product",
        "Once past sign up the home view felt focused",
        "I love how clean the dashboard looks once you are in",
        "Pricing page is confusing — couldn't tell what free included",
        "Free vs paid comparison is hidden behind two clicks",
        "I expected a clear pricing table not marketing copy",
        "Pricing FAQ contradicts the pricing page on discounts",
    ]


def test_cluster_notes_default_config(monkeypatch):
    """Default run with LLM unavailable falls back gracefully."""
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    result = cluster_notes(_three_theme_corpus())
    assert result.notes_clustered == 12
    assert result.cluster_count >= 2
    assert result.llm_used is False
    assert result.llm_backend is None
    # Should have warned about missing keys since use_llm defaults to True
    assert any("no API key" in w for w in result.warnings)


def test_cluster_notes_no_llm_flag(monkeypatch):
    cfg = AffinityConfig(use_llm=False, n_clusters=3, random_state=7)
    result = cluster_notes(_three_theme_corpus(), cfg)
    assert result.llm_used is False
    assert result.cluster_count == 3
    # No LLM-availability warning when use_llm=False
    assert not any("API key" in w for w in result.warnings)


def test_cluster_notes_passes_through_note_objects():
    notes = [Note(id=f"x{i:04d}", text=t, source="p01") for i, t in enumerate(_three_theme_corpus(), start=1)]
    cfg = AffinityConfig(use_llm=False, n_clusters=3, random_state=7)
    result = cluster_notes(notes, cfg)
    # IDs from caller must be preserved
    assert result.notes[0].id == "x0001"
    assert all(n.source == "p01" for n in result.notes)


def test_cluster_notes_too_few_notes_returns_single_bucket():
    cfg = AffinityConfig(use_llm=False, min_clusters=3, max_clusters=4)
    result = cluster_notes(["only one note"], cfg)
    assert result.cluster_count == 1
    assert result.clusters[0].size == 1
    assert any("too small" in w.lower() for w in result.warnings)


def test_cluster_notes_empty_input():
    result = cluster_notes([])
    assert result.notes_clustered == 0
    assert result.clusters == []
    assert result.warnings == ["No notes provided."]


def test_cluster_notes_all_clusters_have_labels():
    cfg = AffinityConfig(use_llm=False, n_clusters=3, random_state=7)
    result = cluster_notes(_three_theme_corpus(), cfg)
    for c in result.clusters:
        assert isinstance(c.label, str) and c.label
        assert isinstance(c.rationale, str) and c.rationale


def test_cluster_notes_to_dict_round_trip_safe():
    cfg = AffinityConfig(use_llm=False, n_clusters=3, random_state=7)
    result = cluster_notes(_three_theme_corpus(), cfg)
    payload = result.to_dict()
    # Top-level keys we promise the CLI will emit
    for key in (
        "notes",
        "clusters",
        "llm_used",
        "llm_backend",
        "notes_clustered",
        "outlier_count",
        "cluster_count",
        "warnings",
        "config",
    ):
        assert key in payload


def test_cluster_notes_outlier_detection_in_pipeline():
    corpus = _three_theme_corpus() + ["zzz qqq vvv bakery croissant"]
    cfg = AffinityConfig(
        use_llm=False,
        n_clusters=3,
        outlier_distance_threshold=0.5,
        random_state=7,
    )
    result = cluster_notes(corpus, cfg)
    assert result.outlier_count >= 1


def test_cluster_notes_warnings_propagate_from_llm_layer(monkeypatch):
    """If use_llm=True but no key is set, the warning should appear."""
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    cfg = AffinityConfig(use_llm=True, n_clusters=3, random_state=7)
    result = cluster_notes(_three_theme_corpus(), cfg)
    assert any("API key" in w for w in result.warnings)


def test_cluster_notes_dedupes_input_strings():
    duplicated = _three_theme_corpus() + _three_theme_corpus()
    cfg = AffinityConfig(use_llm=False, n_clusters=3, random_state=7)
    result = cluster_notes(duplicated, cfg)
    # Deduplicated down to 12
    assert result.notes_clustered == 12
