"""Basic affinity-bot usage example.

Run with::

    python examples/basic_usage.py

The example will use whichever LLM backend has an API key available, or
fall back to the offline rule-based labels if none is configured.
"""

from __future__ import annotations

from affinity_bot import AffinityConfig, cluster_notes

NOTES = [
    # onboarding theme
    "Onboarding flow felt overwhelming — too many steps before I could use the app",
    "I gave up creating an account because there were three different forms",
    "The welcome screen explains nothing about what the product actually does",
    "Sign up took ages and I almost closed the tab",
    # dashboard theme
    "Dashboard cards are pleasantly sized; nothing screams for attention",
    "The dashboard is the best part — finally a tool that doesn't drown me in widgets",
    "I'd pay for the dashboard alone — it respects my time",
    "Once past sign-up, the home view felt calm and focused",
    # pricing theme
    "Pricing page is confusing — I couldn't tell what the free plan included",
    "The free vs paid comparison is hidden behind two clicks",
    "I expected a clear pricing table, not marketing copy",
    "Pricing FAQ contradicts the pricing page on annual discounts",
    # mobile theme
    "Mobile app keeps logging me out — frustrating",
    "On iOS the app loses my session every couple of days",
    "I'd use the mobile version more if it stayed signed in",
    "Phone app and web don't seem to share session — annoying",
    # data export theme
    "I wish there was a way to export my data to CSV",
    "Export options are limited — only PDF, no spreadsheet",
    "No CSV export means I can't pull data into Sheets for analysis",
    "Would love an API endpoint to grab my data programmatically",
]


def main() -> None:
    result = cluster_notes(NOTES, AffinityConfig())

    print(f"\nClustered {result.notes_clustered} notes into "
          f"{result.cluster_count} theme(s) "
          f"(+ {result.outlier_count} outlier note(s))")
    print(f"LLM enriched: {result.llm_used} "
          f"(backend: {result.llm_backend or '—'})\n")

    for cluster in result.clusters:
        marker = "⚠️" if cluster.is_outlier else "•"
        print(f"{marker}  {cluster.label}  ({cluster.size} notes)")
        print(f"    {cluster.rationale}")
        for nid in cluster.note_ids[:3]:
            note = next(n for n in result.notes if n.id == nid)
            print(f"      {nid}  {note.text}")
        if cluster.size > 3:
            print(f"      … and {cluster.size - 3} more")
        print()

    if result.warnings:
        print("Warnings:")
        for w in result.warnings:
            print(f"  · {w}")


if __name__ == "__main__":
    main()
