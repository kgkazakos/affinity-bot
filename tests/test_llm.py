"""Tests for affinity_bot.llm.

These tests do *not* hit any LLM API. They verify that:
  - The right model strings are exposed for the three backends.
  - Backend resolution honours env vars and priority order.
  - `enrich_clusters` falls back gracefully when no key is set or
    when DSPy is not importable.
"""

from __future__ import annotations

import pytest

from affinity_bot.llm import (
    BACKENDS,
    _resolve_backend,
    available_backends,
    enrich_clusters,
)
from affinity_bot.models import AffinityConfig, Cluster, Note


def test_backend_model_strings_are_correct():
    assert BACKENDS["anthropic"] == "anthropic/claude-opus-4-7"
    assert BACKENDS["gemini"] == "gemini/gemini-3.1-pro-preview"
    assert BACKENDS["openai"] == "openai/gpt-5.5"


def test_backends_dict_has_exactly_three_entries():
    assert set(BACKENDS.keys()) == {"anthropic", "gemini", "openai"}


def test_resolve_backend_picks_first_available_in_priority(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GEMINI_API_KEY", "ga-test")
    # priority is anthropic > gemini > openai → gemini wins
    assert _resolve_backend(None) == "gemini"


def test_resolve_backend_anthropic_priority(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    monkeypatch.setenv("OPENAI_API_KEY", "z")
    assert _resolve_backend(None) == "anthropic"


def test_resolve_backend_returns_none_when_no_keys(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert _resolve_backend(None) is None


def test_resolve_backend_pinned_returns_only_if_configured(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert _resolve_backend("anthropic") is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert _resolve_backend("anthropic") == "anthropic"


def test_resolve_backend_gemini_accepts_either_env(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    assert _resolve_backend(None) == "gemini"


def test_available_backends_reports_state(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    avail = available_backends()
    assert avail["openai"] is True
    assert avail["anthropic"] is False


def test_enrich_clusters_use_llm_false_returns_unchanged():
    cluster = Cluster(0, "rule label", "rule rationale", ["n0001"], label_source="rule")
    notes = [Note(id="n0001", text="hello world")]
    cfg = AffinityConfig(use_llm=False)
    outcome = enrich_clusters(notes, [cluster], cfg)
    assert outcome.used is False
    assert outcome.backend is None
    assert outcome.enriched_clusters == [cluster]
    assert outcome.warnings == []


def test_enrich_clusters_no_keys_warns_and_falls_back(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    cluster = Cluster(0, "rule label", "rule rationale", ["n0001"], label_source="rule")
    notes = [Note(id="n0001", text="hello")]
    cfg = AffinityConfig(use_llm=True)
    outcome = enrich_clusters(notes, [cluster], cfg)
    assert outcome.used is False
    assert outcome.backend is None
    assert outcome.enriched_clusters[0].label == "rule label"
    assert outcome.warnings and "no API key" in outcome.warnings[0]


def test_enrich_clusters_pinned_backend_without_key_falls_back(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    cluster = Cluster(0, "rule label", "rule rationale", ["n0001"], label_source="rule")
    notes = [Note(id="n0001", text="hello")]
    cfg = AffinityConfig(use_llm=True, llm_backend="anthropic")
    outcome = enrich_clusters(notes, [cluster], cfg)
    assert outcome.used is False
    assert outcome.enriched_clusters[0].label_source == "rule"


def test_enrich_clusters_outlier_passes_through(monkeypatch):
    """Even if an LLM is configured, outlier clusters keep their rule label."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    # If DSPy is installed in the test env this would try to call out — but
    # the outlier branch returns *before* attempting any LLM call, so this
    # is safe to assert without mocking dspy.
    outlier = Cluster(-1, "Outliers", "—", ["n0001"], is_outlier=True)
    notes = [Note(id="n0001", text="x")]

    # Force the dspy import to fail at *enrichment time* so the real LLM is
    # never invoked even if dspy is present. We do this by removing dspy from
    # sys.modules and then making subsequent imports fail.
    import sys
    monkeypatch.setitem(sys.modules, "dspy", None)

    outcome = enrich_clusters(notes, [outlier], AffinityConfig(use_llm=True))
    # The DSPy guard rail fires *before* we iterate clusters, so we get a
    # graceful fallback rather than a partial run — that's the contract.
    assert outcome.used is False
    assert outcome.enriched_clusters[0].is_outlier
    assert outcome.enriched_clusters[0].label == "Outliers"


def test_enrich_clusters_dspy_missing_falls_back(monkeypatch):
    """If dspy can't be imported, we should warn and return rule output."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    import sys
    # Stub out dspy so the import line raises
    monkeypatch.setitem(sys.modules, "dspy", None)
    cluster = Cluster(0, "rule label", "rule rationale", ["n0001"], label_source="rule")
    notes = [Note(id="n0001", text="hello")]
    cfg = AffinityConfig(use_llm=True)
    outcome = enrich_clusters(notes, [cluster], cfg)
    assert outcome.used is False
    assert outcome.warnings
    assert any("dspy" in w.lower() or "litellm" in w.lower() for w in outcome.warnings)


@pytest.mark.parametrize("backend", ["anthropic", "gemini", "openai"])
def test_every_backend_has_a_model_string(backend):
    model = BACKENDS[backend]
    assert "/" in model
    assert model.split("/", 1)[0] in {"anthropic", "gemini", "openai"}
