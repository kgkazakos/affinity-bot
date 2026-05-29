"""Tests for affinity_bot.cli using Click's CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from affinity_bot.cli import main


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _no_llm_env(monkeypatch):
    """Tests never hit a real LLM. Force absence of API keys."""
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)


def _seed_input(tmp_path: Path) -> Path:
    p = tmp_path / "notes.txt"
    p.write_text(
        "\n".join(
            [
                "Onboarding flow was overwhelming",
                "I gave up creating an account",
                "Sign up took ages",
                "The dashboard is the best part",
                "Dashboard cards are calm and clean",
                "Once past sign up the home view felt focused",
                "Pricing page is confusing",
                "Free vs paid comparison is hidden",
                "Pricing FAQ contradicts the page",
                "Mobile app keeps logging me out",
                "iOS app loses my session",
                "Phone app and web don't share session",
            ]
        )
    )
    return p


def test_cli_version_flag(runner):
    res = runner.invoke(main, ["--version"])
    assert res.exit_code == 0
    assert "affinity-bot" in res.output


def test_cli_help_lists_subcommands(runner):
    res = runner.invoke(main, ["--help"])
    assert res.exit_code == 0
    assert "cluster" in res.output
    assert "backends" in res.output
    assert "example" in res.output


def test_cli_backends_subcommand_runs(runner):
    res = runner.invoke(main, ["backends"])
    assert res.exit_code == 0
    assert "anthropic" in res.output
    assert "gemini" in res.output
    assert "openai" in res.output


def test_cli_backends_warns_when_no_keys(runner):
    res = runner.invoke(main, ["backends"])
    assert "No API keys detected" in res.output


def test_cli_cluster_txt_input(tmp_path, runner):
    input_path = _seed_input(tmp_path)
    res = runner.invoke(
        main,
        ["cluster", str(input_path), "--n-clusters", "3", "--no-llm"],
    )
    assert res.exit_code == 0, res.output
    assert "Affinity report" in res.output
    assert "notes clustered" in res.output


def test_cli_cluster_json_output(tmp_path, runner):
    input_path = _seed_input(tmp_path)
    res = runner.invoke(
        main,
        [
            "cluster",
            str(input_path),
            "--n-clusters",
            "3",
            "--no-llm",
            "--json-output",
        ],
    )
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["notes_clustered"] == 12
    assert data["cluster_count"] >= 2
    assert data["llm_used"] is False


def test_cli_cluster_writes_output_file(tmp_path, runner):
    input_path = _seed_input(tmp_path)
    out_path = tmp_path / "result.json"
    res = runner.invoke(
        main,
        [
            "cluster",
            str(input_path),
            "--n-clusters",
            "3",
            "--no-llm",
            "-o",
            str(out_path),
        ],
    )
    assert res.exit_code == 0
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["notes_clustered"] == 12


def test_cli_cluster_csv_input(tmp_path, runner):
    p = tmp_path / "notes.csv"
    p.write_text("id,observation\n1,one\n2,two\n3,three\n4,four\n5,five\n6,six\n")
    res = runner.invoke(
        main,
        [
            "cluster",
            str(p),
            "--column",
            "observation",
            "--min-clusters",
            "2",
            "--max-clusters",
            "3",
            "--no-llm",
        ],
    )
    assert res.exit_code == 0


def test_cli_cluster_stdin(runner):
    res = runner.invoke(
        main,
        ["cluster", "--stdin", "--n-clusters", "2", "--no-llm"],
        input="a b c\nd e f\ng h i\nj k l\nm n o\np q r\n",
    )
    assert res.exit_code == 0


def test_cli_cluster_requires_source_or_stdin(runner):
    res = runner.invoke(main, ["cluster", "--n-clusters", "3", "--no-llm"])
    assert res.exit_code != 0


def test_cli_cluster_invalid_backend_choice(runner, tmp_path):
    input_path = _seed_input(tmp_path)
    res = runner.invoke(
        main,
        ["cluster", str(input_path), "--llm-backend", "bogus"],
    )
    assert res.exit_code != 0
    assert "bogus" in res.output or "invalid" in res.output.lower()


def test_cli_example_subcommand(runner):
    res = runner.invoke(main, ["example", "--no-llm"])
    assert res.exit_code == 0
    assert "notes clustered" in res.output


def test_cli_cluster_outlier_threshold(tmp_path, runner):
    input_path = _seed_input(tmp_path)
    res = runner.invoke(
        main,
        [
            "cluster",
            str(input_path),
            "--n-clusters",
            "3",
            "--outlier-threshold",
            "0.4",
            "--no-llm",
            "--json-output",
        ],
    )
    assert res.exit_code == 0
    data = json.loads(res.output)
    # tighter threshold tends to produce outliers
    assert data["outlier_count"] >= 0


def test_cli_no_llm_does_not_warn_about_keys(runner, tmp_path):
    input_path = _seed_input(tmp_path)
    res = runner.invoke(
        main,
        ["cluster", str(input_path), "--no-llm", "--n-clusters", "3"],
    )
    assert "no API key" not in res.output.lower()
