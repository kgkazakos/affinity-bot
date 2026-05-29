"""Tests for affinity_bot.clusterer."""

from __future__ import annotations

from affinity_bot.clusterer import cluster, top_terms_for_cluster
from affinity_bot.extract import notes_from_strings
from affinity_bot.models import AffinityConfig


def _three_theme_corpus() -> list[str]:
    return [
        # onboarding theme
        "Onboarding flow felt overwhelming with too many steps",
        "I gave up creating an account because there were three forms",
        "Sign up took ages and I almost closed the tab",
        "The welcome screen explains nothing about the product",
        # dashboard theme
        "Dashboard cards are pleasantly sized and calm",
        "The dashboard is the best part of the product",
        "Once past sign up the home view felt focused",
        "I love how clean the dashboard looks once you are in",
        # pricing theme
        "Pricing page is confusing — couldn't tell what free included",
        "Free vs paid comparison is hidden behind two clicks",
        "I expected a clear pricing table not marketing copy",
        "Pricing FAQ contradicts the pricing page on discounts",
    ]


def test_cluster_returns_assignment_per_note():
    notes = notes_from_strings(_three_theme_corpus())
    cfg = AffinityConfig(n_clusters=3, random_state=7)
    assignment = cluster(notes, cfg)
    assert len(assignment.note_ids) == len(notes)
    assert len(assignment.labels) == len(notes)


def test_cluster_chooses_pinned_k():
    notes = notes_from_strings(_three_theme_corpus())
    cfg = AffinityConfig(n_clusters=4, random_state=7)
    assignment = cluster(notes, cfg)
    real_clusters = {lab for lab in assignment.labels if lab >= 0}
    assert len(real_clusters) == 4


def test_cluster_auto_k_within_bounds():
    notes = notes_from_strings(_three_theme_corpus())
    cfg = AffinityConfig(min_clusters=2, max_clusters=5, random_state=7)
    assignment = cluster(notes, cfg)
    assert cfg.min_clusters <= assignment.chosen_k <= cfg.max_clusters


def test_cluster_groups_similar_notes_together():
    notes = notes_from_strings(_three_theme_corpus())
    cfg = AffinityConfig(n_clusters=3, random_state=7, outlier_distance_threshold=1.0)
    assignment = cluster(notes, cfg)
    labels = dict(zip(assignment.note_ids, assignment.labels, strict=False))
    # onboarding notes (n0001..n0004) should mostly share a label
    onboarding_labs = [labels[f"n{i:04d}"] for i in range(1, 5)]
    pricing_labs = [labels[f"n{i:04d}"] for i in range(9, 13)]
    # majority of each themed block should share the dominant label
    assert max(onboarding_labs.count(lab) for lab in set(onboarding_labs)) >= 3
    assert max(pricing_labs.count(lab) for lab in set(pricing_labs)) >= 3
    # and the dominant onboarding/pricing labels should differ
    onboarding_dom = max(set(onboarding_labs), key=onboarding_labs.count)
    pricing_dom = max(set(pricing_labs), key=pricing_labs.count)
    assert onboarding_dom != pricing_dom


def test_cluster_outliers_when_threshold_strict():
    notes = notes_from_strings(_three_theme_corpus() + ["zzz qqq vvv bakery croissant"])
    cfg = AffinityConfig(n_clusters=3, outlier_distance_threshold=0.5, random_state=7)
    assignment = cluster(notes, cfg)
    # at least the bakery/foreign note should be tagged as outlier under a strict threshold
    assert -1 in assignment.labels


def test_cluster_outliers_disabled_when_threshold_one():
    notes = notes_from_strings(_three_theme_corpus() + ["coffee croissant baguette"])
    cfg = AffinityConfig(n_clusters=3, outlier_distance_threshold=1.0, random_state=7)
    assignment = cluster(notes, cfg)
    assert -1 not in assignment.labels


def test_cluster_empty_input():
    assignment = cluster([], AffinityConfig())
    assert assignment.note_ids == []
    assert assignment.labels == []
    assert assignment.chosen_k == 0


def test_cluster_handles_all_stopwords_corpus():
    notes = notes_from_strings(["the a", "an the", "of the"])
    assignment = cluster(notes, AffinityConfig(min_clusters=2, max_clusters=2))
    # All-stopword corpus → all outliers, but call must not raise
    assert set(assignment.labels) == {-1}


def test_top_terms_returns_distinct_words():
    notes = notes_from_strings(_three_theme_corpus())
    cfg = AffinityConfig(n_clusters=3, random_state=7)
    assignment = cluster(notes, cfg)
    for cid in range(3):
        terms = top_terms_for_cluster(cid, assignment, top_k=5)
        assert all(isinstance(t, str) for t in terms)
        assert len(terms) <= 5


def test_top_terms_invalid_cluster_id_returns_empty():
    notes = notes_from_strings(_three_theme_corpus())
    assignment = cluster(notes, AffinityConfig(n_clusters=3, random_state=7))
    assert top_terms_for_cluster(999, assignment, top_k=3) == []
    assert top_terms_for_cluster(-1, assignment, top_k=3) == []


def test_cluster_deterministic_with_same_seed():
    notes = notes_from_strings(_three_theme_corpus())
    a = cluster(notes, AffinityConfig(n_clusters=3, random_state=42))
    b = cluster(notes, AffinityConfig(n_clusters=3, random_state=42))
    assert a.labels == b.labels
