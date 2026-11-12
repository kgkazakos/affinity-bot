# affinity-bot

**LLM-powered affinity diagramming and theme clustering from raw research notes.**
CPR Orbital Tool **#36** · Season 6 · Pre-study & impact.

![affinity-bot screenshot](docs/affinity-bot-ui.png)

You finished a round of interviews. You have hundreds of observations,
quotes, and field notes scattered across a doc, a CSV, or a wall of sticky
notes. Now comes the part every researcher knows too well: spend a day
clustering them into themes.

`affinity-bot` is a first-pass affinity wall — it groups your notes by
semantic similarity, gives each cluster a researcher-style label, and
flags the outliers that don't fit. The rule-based core (TF-IDF + KMeans
+ top-term labelling) runs **fully offline** with zero API calls. An LLM
enrichment layer is **on by default** across three frontier models and
rewrites the keyword soup into proper noun-phrase theme labels with
one-sentence rationale.

This is a starting point, not a finished thematic analysis. Outliers are
surfaced for you to review, not silently absorbed. The labels are
heuristics, not validated codes. Treat the output the way you'd treat a
junior researcher's first pass: useful, fast, and worth challenging.

---

## Install

```bash
pip install affinity-bot                 # core + CLI + LLM layer
pip install "affinity-bot[ui]"           # + Streamlit UI
```

LLM enrichment is **part of the core install** — `dspy-ai` and `litellm`
are not optional extras. The layer activates when an API key is present
in the environment (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY` /
`GOOGLE_API_KEY`, or `OPENAI_API_KEY`). With no key — or with
`--no-llm` — the tool falls back to the offline rule-based labels and
never hard-fails.

---

## Usage

```bash
# Cluster a file of notes (one per line)
affinity-bot cluster notes.txt

# CSV input — auto-detects a note column, or pin one
affinity-bot cluster observations.csv --column quote

# Pipe from another tool
cat notes.md | affinity-bot cluster --stdin

# Force the fully-offline rule-based path
affinity-bot cluster notes.txt --no-llm

# Pin a backend (default is auto: first available key, anthropic > gemini > openai)
affinity-bot cluster notes.txt --llm-backend gemini

# Machine-readable JSON for CI or downstream tooling
affinity-bot cluster notes.txt --json-output > result.json

# Try the built-in 25-note example
affinity-bot example

# Which backends are configured?
affinity-bot backends
```

In Python:

```python
from affinity_bot import cluster_notes, AffinityConfig

notes = [
    "Sign-up flow was overwhelming",
    "I gave up creating an account",
    "Dashboard finally feels calm",
    # ... more notes
]

result = cluster_notes(notes, AffinityConfig(use_llm=True))
for c in result.clusters:
    print(f"{c.label}  ({c.size} notes)")
    print(f"  {c.rationale}")
```

---

## What this is *not*

This is the honest part, and it matters most:

- **Not a finished thematic analysis.** It's a first-pass affinity wall.
  Reorganise, rename, split, and merge clusters as you would on a real wall.
- **Not validated coding.** Labels are heuristic noun phrases, not codes
  that have been negotiated with a co-researcher or checked for
  inter-rater reliability.
- **Not a substitute for outlier review.** Notes that don't fit are
  surfaced explicitly. They are often the most interesting signal in the
  corpus — read them.
- **Not deterministic across LLM runs.** With the LLM layer on, labels
  vary slightly between calls. With `--no-llm` the output is fully
  deterministic given a fixed seed.

---

## UI

```bash
streamlit run affinity_bot/app.py
```

A drop-zone for pasting notes or uploading a `.txt`/`.csv`/`.json` file,
clustering controls in the sidebar (auto-k or pinned, outlier threshold,
LLM backend), and theme cards with sample notes, keyword chips, and a
JSON export. DM Mono / DM Sans, consistent with the rest of the CPR
Orbital tools.

---

## LLM backends

| Backend   | Model string                    |
|-----------|---------------------------------|
| Anthropic | `anthropic/claude-opus-4-7`     |
| Gemini    | `gemini/gemini-4-argon` |
| OpenAI    | `openai/gpt-5.5`                |

Model strings live in `affinity_bot/llm.py` (`BACKENDS`) — edit them if
your account exposes different names. Resolution order when no backend
is pinned: **anthropic → gemini → openai** (first one with a configured
key wins).

When the LLM layer fires, it generates two things per cluster: a short
noun-phrase label (e.g. *"Sign-up flow friction"*) and a one-sentence
plain-language rationale. Outlier clusters are intentionally skipped —
the LLM is not asked to invent themes for notes that don't belong
together.

---

## How it works

1. **Extract** — parse notes from `.txt`/`.md` (one per line), `.csv`
   (auto-detect column), `.json` (list of strings or objects), or
   stdin. Bullet markers and enclosing quotes are stripped;
   duplicates are removed.
2. **Vectorise** — TF-IDF over unigrams + bigrams, English stopwords,
   capped vocabulary, L2-normalised so KMeans approximates cosine
   clustering.
3. **Choose `k`** — pin with `--n-clusters`, or auto-pick by inertia
   elbow within `[min_clusters, max_clusters]`.
4. **Cluster** — KMeans with a fixed random seed for determinism.
5. **Flag outliers** — notes whose cosine distance to their assigned
   centroid exceeds `--outlier-threshold` are re-tagged to cluster `-1`.
6. **Label** — top-weighted TF-IDF terms become the rule-based label and
   rationale. With the LLM on, those are rewritten into a researcher's
   voice via a DSPy `ChainOfThought` program.

---

## Configuration reference

| Flag | Default | Notes |
|---|---|---|
| `--n-clusters` | auto | Pin `k` explicitly. |
| `--min-clusters` / `--max-clusters` | 3 / 8 | Bounds for auto-`k`. |
| `--outlier-threshold` | 0.85 | Cosine distance above which a note becomes an outlier. Set to 1.0 to disable. |
| `--keywords-per-cluster` | 4 | How many TF-IDF terms appear in the keyword signature. |
| `--llm-backend` | auto | Pin to `anthropic`, `gemini`, or `openai`. |
| `--llm-model` | — | Override the model string for the chosen backend. |
| `--no-llm` | off | Disable LLM enrichment; keep rule-based labels. |
| `--seed` | 7 | Random seed for KMeans. |
| `-o`, `--output` | — | Write a JSON result file alongside the terminal report. |
| `--json-output` | off | Print machine-readable JSON to stdout. |
| `--stdin` | off | Read notes from stdin instead of a file. |

---

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

The suite covers models, extraction across formats, clustering,
labelling, LLM fallback paths (without ever hitting an API), end-to-end
pipeline, and the CLI. Targets Python 3.10 / 3.11 / 3.12 — same as the
rest of the series.

---

## Part of the CPR Ecosystem

`affinity-bot` is **tool #36 of 40** in the **CPR Orbital Series** — a
near-daily release cadence of open-source tools for AI-native product
research infrastructure, running through December 2026. The broader
intellectual frame is **Computational Product Research (CPR)**.

It pairs naturally with:

- [`transcript-cleaner`](https://github.com/kgkazakos/transcript-cleaner)
  (#01) — clean up your raw interview transcripts before extracting
  notes.
- [`pii-scrubber`](https://github.com/kgkazakos/pii-scrubber)
  (#35) — de-identify notes before clustering.
- [`jtbd-extractor`](https://github.com/kgkazakos/jtbd-extractor)
  (#37, coming soon) — pull Jobs-to-Be-Done statements from interview
  transcripts; feed those statements into `affinity-bot` for a
  second-pass thematic view.
- [`stakeholder-brief-writer`](https://github.com/kgkazakos/stakeholder-brief-writer)
  (#38, coming soon) — turn a labelled set of themes into a one-page
  executive summary.

Roadmap & series overview: [kgkazakos.github.io](https://kgkazakos.github.io)

---

## License

MIT © Kostas Kazakos

This is independent CPR research, released under a permissive license.
Not affiliated with any employer.
