"""Click-based CLI for affinity-bot."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from affinity_bot import __version__
from affinity_bot.extract import (
    notes_from_csv,
    notes_from_json,
    notes_from_path,
    notes_from_text,
)
from affinity_bot.llm import BACKENDS, available_backends
from affinity_bot.models import AffinityConfig
from affinity_bot.pipeline import cluster_notes

console = Console()


@click.group()
@click.version_option(__version__, prog_name="affinity-bot")
def main() -> None:
    """affinity-bot — LLM-powered affinity diagramming for research notes.

    The rule-based core runs fully offline. LLM enrichment is on by default
    and gracefully falls back when no API key is present.
    """


@main.command("cluster")
@click.argument(
    "source",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
)
@click.option(
    "--column",
    type=str,
    default=None,
    help="CSV column containing notes (auto-detected if omitted).",
)
@click.option(
    "--n-clusters",
    type=int,
    default=None,
    help="Pin the number of clusters. Default: auto-pick.",
)
@click.option(
    "--min-clusters",
    type=int,
    default=3,
    show_default=True,
    help="Lower bound when auto-picking k.",
)
@click.option(
    "--max-clusters",
    type=int,
    default=8,
    show_default=True,
    help="Upper bound when auto-picking k.",
)
@click.option(
    "--outlier-threshold",
    type=float,
    default=0.85,
    show_default=True,
    help="Cosine distance above which a note is tagged as an outlier.",
)
@click.option(
    "--keywords-per-cluster",
    type=int,
    default=4,
    show_default=True,
    help="Top TF-IDF terms shown in the keyword signature.",
)
@click.option(
    "--llm-backend",
    type=click.Choice(["anthropic", "gemini", "openai"]),
    default=None,
    help="Pin the LLM backend. Default: first one with a configured key.",
)
@click.option(
    "--llm-model",
    type=str,
    default=None,
    help="Override the model string for the chosen backend.",
)
@click.option(
    "--no-llm",
    is_flag=True,
    default=False,
    help="Disable LLM enrichment; keep the rule-based labels.",
)
@click.option(
    "--seed",
    type=int,
    default=7,
    show_default=True,
    help="Random seed for KMeans initialisation.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write a JSON result file in addition to the terminal report.",
)
@click.option(
    "--json-output",
    is_flag=True,
    default=False,
    help="Print machine-readable JSON to stdout (suitable for piping).",
)
@click.option(
    "--stdin",
    "from_stdin",
    is_flag=True,
    default=False,
    help="Read notes from stdin (one per line) instead of a file.",
)
def cluster_cmd(
    source: Optional[Path],
    column: Optional[str],
    n_clusters: Optional[int],
    min_clusters: int,
    max_clusters: int,
    outlier_threshold: float,
    keywords_per_cluster: int,
    llm_backend: Optional[str],
    llm_model: Optional[str],
    no_llm: bool,
    seed: int,
    output: Optional[Path],
    json_output: bool,
    from_stdin: bool,
) -> None:
    """Cluster a file (or stdin) of research notes into affinity themes."""
    notes = _read_notes(source, column, from_stdin)
    if not notes:
        console.print("[red]No notes found in input.[/red]")
        sys.exit(2)

    cfg = AffinityConfig(
        n_clusters=n_clusters,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        outlier_distance_threshold=outlier_threshold,
        keywords_per_cluster=keywords_per_cluster,
        use_llm=not no_llm,
        llm_backend=llm_backend,
        llm_model=llm_model,
        random_state=seed,
    )

    result = cluster_notes(notes, cfg)

    if output is not None:
        output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    if json_output:
        click.echo(json.dumps(result.to_dict(), indent=2))
        return

    _render_terminal_report(result)


@main.command("backends")
def backends_cmd() -> None:
    """Show which LLM backends are configured (API key detected)."""
    table = Table(title="affinity-bot · LLM backends", show_header=True, header_style="bold")
    table.add_column("Backend", style="bold")
    table.add_column("Model")
    table.add_column("Configured")
    avail = available_backends()
    for backend, model in BACKENDS.items():
        configured = avail.get(backend, False)
        marker = "[green]yes[/green]" if configured else "[dim]no[/dim]"
        table.add_row(backend, model, marker)
    console.print(table)
    if not any(avail.values()):
        console.print(
            "[yellow]No API keys detected.[/yellow] Set "
            "ANTHROPIC_API_KEY, GEMINI_API_KEY (or GOOGLE_API_KEY), or "
            "OPENAI_API_KEY to enable LLM enrichment, or run with --no-llm "
            "for the offline rule-based path."
        )


@main.command("example")
@click.option(
    "--no-llm",
    is_flag=True,
    default=False,
    help="Run the example with the rule-based path only.",
)
def example_cmd(no_llm: bool) -> None:
    """Run affinity-bot on a built-in example corpus of research notes."""
    notes = _example_notes()
    cfg = AffinityConfig(use_llm=not no_llm)
    result = cluster_notes(notes, cfg)
    _render_terminal_report(result)


def _read_notes(
    source: Optional[Path], column: Optional[str], from_stdin: bool
) -> list:
    if from_stdin:
        return notes_from_text(sys.stdin.read(), source="<stdin>")
    if source is None:
        raise click.UsageError("Provide a file path, or use --stdin.")
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return notes_from_csv(source, column=column)
    if suffix == ".json":
        return notes_from_json(source)
    return notes_from_path(source, column=column)


def _render_terminal_report(result) -> None:
    """Render the affinity result to the terminal with Rich."""
    header_lines = [
        f"[bold]affinity-bot[/bold] · v{__version__}",
        f"notes clustered: [bold]{result.notes_clustered}[/bold]   "
        f"clusters: [bold]{result.cluster_count}[/bold]   "
        f"outliers: [bold]{result.outlier_count}[/bold]",
    ]
    if result.llm_used:
        header_lines.append(
            f"LLM: [green]enriched[/green] via [bold]{result.llm_backend}[/bold]"
        )
    elif result.config.use_llm:
        header_lines.append("LLM: [yellow]requested, falling back to rule-based[/yellow]")
    else:
        header_lines.append("LLM: [dim]disabled (rule-based labels)[/dim]")
    console.print(Panel("\n".join(header_lines), title="Affinity report", border_style="cyan"))

    for cluster in result.clusters:
        if cluster.is_outlier and cluster.size == 0:
            continue
        _render_cluster(cluster, result.notes)

    if result.warnings:
        warn_text = Text("\n".join(result.warnings), style="yellow")
        console.print(Panel(warn_text, title="Warnings", border_style="yellow"))


def _render_cluster(cluster, notes_list) -> None:
    notes_by_id = {n.id: n for n in notes_list}
    header = Text()
    header.append(cluster.label, style="bold")
    header.append(f"   {cluster.size} note(s)", style="dim")
    if cluster.label_source == "llm":
        header.append("   ·  llm", style="green")
    elif cluster.is_outlier:
        header.append("   ·  outliers", style="red")
    else:
        header.append("   ·  rule", style="dim")

    body = Text()
    body.append(cluster.rationale + "\n\n", style="italic")
    if cluster.keyword_signature:
        body.append("keywords: ", style="dim")
        body.append(", ".join(cluster.keyword_signature) + "\n\n", style="cyan")
    for nid in cluster.note_ids[:6]:
        note = notes_by_id.get(nid)
        if note is None:
            continue
        body.append(f"  {nid} ", style="dim")
        body.append(note.text + "\n")
    if cluster.size > 6:
        body.append(f"  … and {cluster.size - 6} more note(s)\n", style="dim")

    style = "red" if cluster.is_outlier else "white"
    console.print(Panel(body, title=header, border_style=style))


def _example_notes() -> list[str]:
    return [
        "Onboarding flow felt overwhelming — too many steps before I could use the app",
        "I gave up creating an account because there were three different forms",
        "The welcome screen explains nothing about what the product actually does",
        "I expected to start using it immediately, instead I had to verify by email twice",
        "Sign up took ages and I almost closed the tab",
        "I love how clean the dashboard looks once you're in",
        "The dashboard is the best part — finally a tool that doesn't drown me in widgets",
        "Once past sign-up, the home view felt calm and focused",
        "Dashboard cards are pleasantly sized; nothing screams for attention",
        "I'd pay for the dashboard alone — it respects my time",
        "Pricing page is confusing — I couldn't tell what the free plan included",
        "The free vs paid comparison is hidden behind two clicks",
        "I expected a clear pricing table, not marketing copy",
        "Couldn't figure out what 'fair use' meant for free tier",
        "Pricing FAQ contradicts the pricing page on annual discounts",
        "Mobile app keeps logging me out — frustrating",
        "On iOS the app loses my session every couple of days",
        "I'd use the mobile version more if it stayed signed in",
        "Mobile login flow is buggy — face ID never works first try",
        "Phone app and web don't seem to share session — annoying",
        "I wish there was a way to export my data to CSV",
        "Export options are limited — only PDF, no spreadsheet",
        "No CSV export means I can't pull data into Sheets for analysis",
        "Would love an API endpoint to grab my data programmatically",
        "Backup-to-disk would be a killer feature for power users",
    ]


if __name__ == "__main__":
    main()
