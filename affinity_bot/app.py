"""Streamlit UI for affinity-bot.

Run with::

    streamlit run affinity_bot/app.py

The UI mirrors the CLI: drop in raw notes, choose clustering parameters,
optionally pin an LLM backend, then explore the resulting themes.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from affinity_bot import __version__
from affinity_bot.extract import notes_from_csv, notes_from_json, notes_from_text
from affinity_bot.llm import BACKENDS, available_backends
from affinity_bot.models import AffinityConfig
from affinity_bot.pipeline import cluster_notes

st.set_page_config(
    page_title="affinity-bot",
    page_icon="🪡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _escape(text: Any) -> str:
    if not isinstance(text, str):
        text = str(text)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

# ---------- styling ----------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;700&display=swap');
      html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
      }
      .stApp { background: #fbfbf8; }
      h1, h2, h3, h4 { font-family: 'DM Sans', sans-serif; letter-spacing: -0.01em; }
      code, .stCode, .mono { font-family: 'DM Mono', monospace !important; }
      .stat-card {
        background: #ffffff;
        border: 1px solid #ececea;
        border-radius: 10px;
        padding: 14px 16px;
        height: 100%;
      }
      .stat-num { font-size: 26px; font-weight: 700; color: #1f2330; font-family: 'DM Mono', monospace;}
      .stat-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #707686; }
      .cluster-card {
        background: #ffffff;
        border: 1px solid #ececea;
        border-left: 3px solid #d8731f;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
      }
      .cluster-card.outlier {
        border-left-color: #b13a3a;
        background: #fff8f7;
      }
      .cluster-card.rule {
        border-left-color: #9aa0ad;
      }
      .cluster-label {
        font-weight: 700; font-size: 16px; color: #1f2330;
      }
      .cluster-meta {
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        color: #707686;
        margin-bottom: 6px;
      }
      .cluster-rationale {
        color: #3a3f4d; font-size: 14px; margin-bottom: 10px; line-height: 1.5;
      }
      .note-line {
        font-size: 13px;
        color: #2c3140;
        padding: 4px 0;
        border-bottom: 1px dashed #f1efea;
      }
      .note-id { color: #a09a8e; font-family: 'DM Mono', monospace; font-size: 11px; margin-right: 6px;}
      .keyword-chip {
        display: inline-block;
        background: #f3efe6;
        color: #6e5a2a;
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 99px;
        margin-right: 5px;
        margin-bottom: 3px;
      }
      .footer-line {
        color: #9aa0ad; font-family: 'DM Mono', monospace; font-size: 11px;
        margin-top: 32px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- sidebar ----------
with st.sidebar:
    st.markdown(
        "<div style='font-family: DM Mono, monospace; font-size:11px; color:#9aa0ad; "
        "letter-spacing:0.08em; text-transform:uppercase;'>CPR Orbital · #36</div>",
        unsafe_allow_html=True,
    )
    st.markdown("### affinity-bot")
    st.caption(f"v{__version__}")
    st.markdown("---")

    st.markdown("#### Clustering")
    auto_k = st.checkbox("Auto-pick number of clusters", value=True)
    if auto_k:
        n_clusters = None
        c1, c2 = st.columns(2)
        with c1:
            min_clusters = st.number_input("min", 2, 12, 3)
        with c2:
            max_clusters = st.number_input("max", 2, 20, 8)
    else:
        n_clusters = st.number_input("number of clusters", 2, 20, 5)
        min_clusters, max_clusters = 3, 8
    outlier_threshold = st.slider(
        "outlier threshold (cosine dist.)",
        min_value=0.5,
        max_value=1.0,
        value=0.85,
        step=0.01,
    )
    keywords_per_cluster = st.number_input("keywords per cluster", 1, 10, 4)

    st.markdown("#### LLM enrichment")
    use_llm = st.checkbox("Enabled (on by default)", value=True)
    backend_options = ["auto"] + list(BACKENDS.keys())
    backend_choice = st.selectbox(
        "backend",
        backend_options,
        index=0,
        help="Auto: tries anthropic → gemini → openai based on configured keys.",
        disabled=not use_llm,
    )
    llm_backend = None if backend_choice == "auto" else backend_choice

    with st.expander("backend status"):
        avail = available_backends()
        for backend, model in BACKENDS.items():
            mark = "✓" if avail.get(backend) else "—"
            st.markdown(
                f"<span class='mono'>{mark} {backend}: {model}</span>",
                unsafe_allow_html=True,
            )

# ---------- header ----------
st.markdown(
    "<h1 style='margin-bottom:4px;'>affinity-bot</h1>"
    "<div style='color:#707686; margin-bottom: 18px;'>Cluster raw research notes into "
    "themes. A first-pass affinity wall, not a finished thematic analysis.</div>",
    unsafe_allow_html=True,
)

# ---------- input ----------
tab_paste, tab_upload, tab_example = st.tabs(["Paste notes", "Upload file", "Run example"])

raw_notes: list[str] | None = None
input_label = ""

with tab_paste:
    pasted = st.text_area(
        "One note per line",
        height=240,
        placeholder=(
            "Sign-up flow was overwhelming\n"
            "Dashboard finally feels calm\n"
            "Pricing page is confusing\n"
            "Mobile app keeps logging me out\n"
            "I wish there was a CSV export option"
        ),
        key="paste",
    )
    if pasted.strip():
        raw_notes = notes_from_text(pasted, source="<paste>")
        input_label = "pasted notes"

with tab_upload:
    uploaded = st.file_uploader(
        "Drop a .txt / .md / .csv / .json file",
        type=["txt", "md", "csv", "json"],
        key="upload",
    )
    column_hint = st.text_input(
        "CSV column name (optional — auto-detected from common names)",
        value="",
    )
    if uploaded is not None:
        data = uploaded.read().decode("utf-8")
        name = uploaded.name.lower()
        if name.endswith(".csv"):
            raw_notes = notes_from_csv(data, column=column_hint or None, source=uploaded.name)
        elif name.endswith(".json"):
            raw_notes = notes_from_json(data, source=uploaded.name)
        else:
            raw_notes = notes_from_text(data, source=uploaded.name)
        input_label = uploaded.name

with tab_example:
    if st.button("Load 25-note example"):
        from affinity_bot.cli import _example_notes
        raw_notes = notes_from_text("\n".join(_example_notes()), source="<example>")
        input_label = "built-in example"
        st.session_state["example_loaded"] = True

if raw_notes is not None and raw_notes:
    st.caption(f"{len(raw_notes)} note(s) from **{input_label}**")

# ---------- run ----------
go = st.button(
    "Cluster notes →",
    type="primary",
    disabled=raw_notes is None or not raw_notes,
)

if go and raw_notes:
    cfg = AffinityConfig(
        n_clusters=int(n_clusters) if n_clusters is not None else None,
        min_clusters=int(min_clusters),
        max_clusters=int(max_clusters),
        outlier_distance_threshold=float(outlier_threshold),
        keywords_per_cluster=int(keywords_per_cluster),
        use_llm=use_llm,
        llm_backend=llm_backend,
    )

    with st.spinner("Vectorising, clustering, labelling…"):
        result = cluster_notes(raw_notes, cfg)

    # Stat cards
    s1, s2, s3, s4 = st.columns(4)
    for col, num, label in (
        (s1, result.notes_clustered, "notes"),
        (s2, result.cluster_count, "themes"),
        (s3, result.outlier_count, "outliers"),
        (s4, result.llm_backend or ("rule" if not use_llm else "fallback"), "label source"),
    ):
        col.markdown(
            f"<div class='stat-card'><div class='stat-num'>{num}</div>"
            f"<div class='stat-label'>{label}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Warnings
    if result.warnings:
        for w in result.warnings:
            st.warning(w)

    # Cluster cards
    notes_by_id = {n.id: n for n in result.notes}
    for cluster in result.clusters:
        css_class = "cluster-card outlier" if cluster.is_outlier else (
            "cluster-card rule" if cluster.label_source == "rule" else "cluster-card"
        )
        meta_bits = [f"#{cluster.cluster_id}", f"{cluster.size} note(s)"]
        meta_bits.append("outliers" if cluster.is_outlier else f"label · {cluster.label_source}")
        meta_line = "   ·   ".join(meta_bits)

        kw_html = "".join(
            f"<span class='keyword-chip'>{kw}</span>"
            for kw in cluster.keyword_signature[:cfg.keywords_per_cluster]
        )

        note_lines = []
        for nid in cluster.note_ids[:8]:
            note = notes_by_id.get(nid)
            if note is None:
                continue
            note_lines.append(
                f"<div class='note-line'><span class='note-id'>{nid}</span>{_escape(note.text)}</div>"
            )
        if cluster.size > 8:
            note_lines.append(
                f"<div class='note-line' style='color:#9aa0ad;'>… and {cluster.size - 8} more</div>"
            )

        st.markdown(
            f"""
            <div class='{css_class}'>
              <div class='cluster-meta'>{meta_line}</div>
              <div class='cluster-label'>{_escape(cluster.label)}</div>
              <div class='cluster-rationale'>{_escape(cluster.rationale)}</div>
              <div style='margin-bottom:8px;'>{kw_html}</div>
              {''.join(note_lines)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # JSON export
    with st.expander("JSON export"):
        payload = json.dumps(result.to_dict(), indent=2)
        st.download_button(
            "Download affinity_result.json",
            payload,
            file_name="affinity_result.json",
            mime="application/json",
        )
        st.code(payload, language="json")

st.markdown(
    f"<div class='footer-line'>affinity-bot · v{__version__} · CPR Orbital #36 · MIT · "
    f"<a href='https://github.com/kgkazakos' style='color:inherit;'>kgkazakos</a></div>",
    unsafe_allow_html=True,
)
