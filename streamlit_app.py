"""
Self-Healing Big Data Pipeline Agent (AWS) — Streamlit UI

Run locally:
    streamlit run streamlit_app.py

Deploy to Streamlit Cloud:
    Push to GitHub, connect at share.streamlit.io
    Add secrets: OPENAI_API_KEY (or GEMINI_API_KEY + LLM_PROVIDER=gemini), PINECONE_API_KEY,
    PINECONE_INDEX_NAME, GITHUB_TOKEN
"""

import os
import queue
import logging
import datetime
import threading
import time

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap — env vars must be set BEFORE any project imports
# ---------------------------------------------------------------------------

load_dotenv()

try:
    for key, val in st.secrets.items():
        if not os.getenv(key):
            os.environ[key] = str(val)
except Exception:
    pass

from utils.llm_defaults import nl_api_key as _nl_api_key, using_gemini as _using_gemini  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Self-Healing Big Data Pipeline Agent",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark cloud / tech theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>

/* ── GLOBAL ─────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(160deg, #060c1a 0%, #0d1b35 45%, #070e20 100%);
    color: #e2e8f0;
}
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 3rem;
}

/* ── TOP HEADER BAR ──────────────────────────────────────── */
[data-testid="stHeader"] {
    background: rgba(6, 12, 26, 0.97) !important;
    border-bottom: 1px solid rgba(56, 189, 248, 0.10) !important;
    backdrop-filter: blur(16px) !important;
}
/* Hide the rainbow decoration line */
[data-testid="stDecoration"],
[data-testid="stDecorationTop"] {
    display: none !important;
}
/* Hamburger / nav icons */
[data-testid="stHeader"] button,
[data-testid="stHeader"] [data-testid="collapsedControl"] {
    color: #64748b !important;
    border: none !important;
    background: transparent !important;
}
[data-testid="stHeader"] button:hover {
    color: #38bdf8 !important;
    background: rgba(56, 189, 248, 0.1) !important;
    border-radius: 8px !important;
}
/* Deploy button — make it fit the theme */
[data-testid="stToolbar"] {
    background: transparent !important;
}
.stDeployButton > button,
[data-testid="stDeployButton"] > button {
    background: rgba(14, 165, 233, 0.10) !important;
    border: 1px solid rgba(56, 189, 248, 0.30) !important;
    color: #38bdf8 !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 0.25rem 0.75rem !important;
}
.stDeployButton > button:hover,
[data-testid="stDeployButton"] > button:hover {
    background: rgba(14, 165, 233, 0.20) !important;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.25) !important;
}
/* Toolbar overflow menu icons */
[data-testid="stToolbarActions"] button {
    color: #475569 !important;
    background: transparent !important;
}
[data-testid="stToolbarActions"] button:hover {
    color: #38bdf8 !important;
    background: rgba(56, 189, 248, 0.08) !important;
    border-radius: 6px !important;
}

/* ── STICKY RUN TABS (Log/Graph) — scoped via the .sticky-tabs-anchor span ── */
/* Collapse the empty anchor's own container so it adds no vertical gap. */
[data-testid="stElementContainer"]:has(.sticky-tabs-anchor),
.element-container:has(.sticky-tabs-anchor) {
    display: none !important;
}
/* WHY this is needed: the run tabs live inside st.status, which is an EXPANDER whose
   body clips overflow — and a clipping ancestor neutralises position:sticky (the bar
   ends up sticky relative to that short box, so it scrolls away). Clear overflow on the
   anchor's whole ancestor chain (scoped: only the run-status subtree carries the anchor)
   AND on the sibling tabs wrappers, so the bar can pin to the viewport instead. */
div:has(.sticky-tabs-anchor),
details:has(.sticky-tabs-anchor),
[data-testid="stExpanderDetails"]:has(.sticky-tabs-anchor),
div:has(.sticky-tabs-anchor) ~ div,
div:has(.sticky-tabs-anchor) ~ div [data-testid="stTabs"],
div:has(.sticky-tabs-anchor) ~ div [data-baseweb="tabs"] {
    overflow: visible !important;
}
/* Pin ONLY the tab bar that FOLLOWS the anchor. The top-level navigation tabs
   precede the anchor in the DOM, so the general-sibling combinator never reaches
   them — they keep scrolling normally. */
div:has(.sticky-tabs-anchor) ~ div [data-baseweb="tab-list"] {
    position: sticky !important;
    top: 3.5rem;                       /* pin just below the fixed Streamlit header */
    z-index: 50;
    background: rgba(6, 12, 26, 0.98);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(56, 189, 248, 0.12);
}

/* ── SIDEBAR ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(10, 18, 40, 0.97) !important;
    border-right: 1px solid rgba(56, 189, 248, 0.18);
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #94a3b8 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e2e8f0 !important;
}

/* ── HEADINGS ────────────────────────────────────────────── */
h1, h2, h3 { color: #e2e8f0 !important; }
p, li { color: #cbd5e1 !important; }

/* ── METRIC CARDS ────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: rgba(30, 41, 59, 0.7) !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    border-radius: 14px !important;
    padding: 1.1rem 1.3rem !important;
    backdrop-filter: blur(12px);
}
[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-weight: 700 !important;
    font-size: 1.4rem !important;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── TEXT AREA ───────────────────────────────────────────── */
.stTextArea textarea {
    background: rgba(10, 18, 40, 0.85) !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 15px !important;
    line-height: 1.6;
}
.stTextArea textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
}
.stTextArea label { color: #94a3b8 !important; }

/* ── BUTTONS ─────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1d4ed8 0%, #0ea5e9 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 12px rgba(14, 165, 233, 0.3);
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.45) !important;
}
.stButton > button[kind="secondary"] {
    background: rgba(30, 41, 59, 0.7) !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    border-radius: 10px !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #38bdf8 !important;
    color: #38bdf8 !important;
}

/* ── CODE BLOCKS (agent log) ─────────────────────────────── */
[data-testid="stCode"] {
    background: rgba(6, 12, 26, 0.9) !important;
    border: 1px solid rgba(56, 189, 248, 0.18) !important;
    border-radius: 12px !important;
}
[data-testid="stCode"] code, [data-testid="stCode"] pre {
    color: #7dd3fc !important;
    font-size: 13px !important;
    line-height: 1.7 !important;
}

/* ── STATUS WIDGET ───────────────────────────────────────── */
[data-testid="stStatus"] {
    background: rgba(10, 18, 40, 0.8) !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    border-radius: 14px !important;
}

/* ── TABS ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(10, 18, 40, 0.6) !important;
    border-radius: 12px !important;
    padding: 6px !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
    gap: 8px;                       /* more space between tabs */
    margin-bottom: 1.2rem !important;  /* air between the tab bar and the content panel */
}
.stTabs [data-baseweb="tab"] {
    color: #64748b !important;
    border-radius: 9px !important;
    font-weight: 500;
    padding: 0.55rem 1.1rem !important;  /* internal breathing room per tab */
}
.stTabs [aria-selected="true"] {
    background: rgba(56, 189, 248, 0.15) !important;
    color: #38bdf8 !important;
}
/* a touch of air at the top of the selected tab's content */
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 0.5rem !important;
}

/* Hide Streamlit's own "Deploy" button (framework chrome — it publishes the APP to Streamlit
   Cloud, unrelated to our pipeline deploy; off-brand for a demo). The ⋮ menu (Rerun/Settings)
   stays for development. */
[data-testid="stDeployButton"],
.stDeployButton { display: none !important; }

/* ── SELECTBOX — trigger ─────────────────────────────────── */
.stSelectbox [data-baseweb="select"] > div {
    background: rgba(10, 18, 40, 0.85) !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] div {
    color: #e2e8f0 !important;
}
.stSelectbox label { color: #94a3b8 !important; }
/* ── SELECTBOX — dropdown popup ──────────────────────────── */
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="menu"] {
    background: #0d1b35 !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6) !important;
}
[data-baseweb="option"] {
    background: transparent !important;
    color: #cbd5e1 !important;
    border-radius: 8px !important;
    padding: 0.55rem 0.9rem !important;
}
[data-baseweb="option"]:hover,
[data-baseweb="option"][aria-selected="true"] {
    background: rgba(56, 189, 248, 0.12) !important;
    color: #38bdf8 !important;
}
li[role="option"] {
    background: transparent !important;
    color: #cbd5e1 !important;
}
li[role="option"]:hover {
    background: rgba(56, 189, 248, 0.12) !important;
    color: #38bdf8 !important;
}

/* ── DIVIDER ─────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid rgba(56, 189, 248, 0.15) !important;
    margin: 1.5rem 0 !important;
}

/* ── ALERTS ──────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
}

/* ── SPINNER ─────────────────────────────────────────────── */
.stSpinner > div { border-top-color: #38bdf8 !important; }

/* ── SELECTBOX — dropdown popup (portal-rendered by BaseUI) ─ */
ul[role="listbox"] {
    background-color: #0d1b35 !important;
    border: 1px solid rgba(56,189,248,0.25) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.7) !important;
    padding: 4px !important;
}
ul[role="listbox"] li,
ul[role="listbox"] li * {
    background-color: transparent !important;
    color: #cbd5e1 !important;
    font-size: 0.9rem !important;
}
ul[role="listbox"] li:hover,
ul[role="listbox"] li:hover * {
    background-color: rgba(56,189,248,0.12) !important;
    color: #38bdf8 !important;
    border-radius: 8px !important;
}
ul[role="listbox"] li[aria-selected="true"],
ul[role="listbox"] li[aria-selected="true"] * {
    background-color: rgba(56,189,248,0.18) !important;
    color: #38bdf8 !important;
}
/* also catch data-baseweb portals */
[data-baseweb="menu"],
[data-baseweb="menu"] ul,
[data-baseweb="list"] {
    background-color: #0d1b35 !important;
    border-radius: 12px !important;
}
[data-baseweb="option"] {
    background-color: transparent !important;
    color: #cbd5e1 !important;
}
[data-baseweb="option"]:hover {
    background-color: rgba(56,189,248,0.12) !important;
    color: #38bdf8 !important;
}

/* ── ARCHITECTURE REPORT CARDS ──────────────────────────── */
.arch-card {
    background: rgba(10,18,40,0.75);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    height: 100%;
    transition: border-color 0.2s;
}
.arch-card-rec {
    border: 2px solid #4ade80 !important;
    box-shadow: 0 0 20px rgba(74,222,128,0.12);
}
.arch-card-plain { border: 1px solid rgba(56,189,248,0.18); }
.arch-card-selected { border: 2px solid #38bdf8 !important; }
.arch-cloud-name {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 0.15rem;
}
.arch-price {
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 0.5rem;
    line-height: 1;
}
.arch-label {
    font-size: 0.78rem;
    color: #64748b;
    margin: 0 0 0.7rem;
    font-style: italic;
}
.arch-pro  { color: #4ade80; font-size: 0.8rem; margin: 0.15rem 0; }
.arch-con  { color: #f87171; font-size: 0.8rem; margin: 0.15rem 0; }
.arch-rec-badge {
    background: rgba(74,222,128,0.12);
    border: 1px solid rgba(74,222,128,0.3);
    border-radius: 6px;
    color: #4ade80;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.1rem 0.5rem;
    display: inline-block;
    margin-bottom: 0.5rem;
}

/* ── SIDEBAR STATUS CARDS ────────────────────────────────── */
.sb-card {
    background: rgba(10,18,40,0.65);
    border: 1px solid rgba(56,189,248,0.13);
    border-radius: 10px;
    padding: 0.55rem 0.85rem 0.35rem;
    margin-bottom: 0.75rem;
}
.sb-title {
    font-size: 0.65rem !important;
    color: #475569 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 0.45rem !important;
}
.sb-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.32rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.82rem;
}
.sb-row:last-child { border-bottom: none; }
.sb-lbl { color: #94a3b8; }
.sb-ok  { color: #4ade80; font-weight: 600; }
.sb-err { color: #f87171; font-weight: 600; }
.sb-warn-card {
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.2);
    border-radius: 10px;
    padding: 0.6rem 0.85rem;
    margin-bottom: 0.75rem;
}
.sb-warn-title { color: #fbbf24; font-size: 0.82rem; font-weight: 600; margin: 0 0 0.25rem; }
.sb-warn-body  { color: #64748b; font-size: 0.74rem; line-height: 1.55; margin: 0; }
.sb-warn-body code { color: #7dd3fc; }

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Lazy graph loader
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading agent graph...")
def get_graph():
    from graph import app
    return app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NODE_LABELS = {
    "supervisor":    "🎯 SUPERVISOR",
    "architect":     "🏗️  ARCHITECT",
    "infra":         "⚙️  INFRA",
    "medic":         "🏥 MEDIC",
    "execute_tools": "🔧 TOOLS",
}
_CLOUD_FLAGS = {"aws": "🟠 AWS"}

# ---------------------------------------------------------------------------
# Pipeline visualization helpers
# ---------------------------------------------------------------------------

_PIPELINE_NODES = [
    ("supervisor", "🎯 SUPERVISOR"),
    ("architect",  "🏗️ ARCHITECT"),
    ("infra",      "⚙️ INFRA"),
    ("medic",      "🏥 MEDIC"),
]
def _legend_swatch(color: str, label: str) -> str:
    return (f'<span style="display:inline-flex;align-items:center;gap:0.35rem;">'
            f'<span style="width:13px;height:13px;border-radius:4px;background:#0b1220;'
            f'border:2px solid {color};display:inline-block;"></span>{label}</span>')


# Color key for the graph, matching the node stroke colors (a convention every execution-graph
# UI carries — n8n / React Flow / LangSmith). Static, rendered once below the graph.
_GRAPH_LEGEND_HTML = (
    '<div style="display:flex;flex-wrap:wrap;gap:0.55rem 1.15rem;align-items:center;'
    'margin:0.6rem 0 0;font-size:0.74rem;color:#94a3b8;">'
    + _legend_swatch("#38bdf8", "Active (running)")
    + _legend_swatch("#64748b", "Supervisor · hub")
    + _legend_swatch("#4ade80", "Completed ✓")
    + _legend_swatch("#f87171", "Failed ✗")
    + _legend_swatch("#fbbf24", "Unverified !")
    + _legend_swatch("#334155", "Pending")
    + '<span style="display:inline-flex;align-items:center;gap:0.35rem;">'
      '<span style="width:22px;border-top:2px dashed #38bdf8;display:inline-block;"></span>'
      'current transition</span>'
    + '<span style="display:inline-flex;align-items:center;gap:0.35rem;color:#64748b;">'
      '<b style="color:#cbd5e1;">×N</b> times a node ran</span>'
    + '</div>'
)


def _render_agent_svg(node_statuses: dict, last_edge, run_counts: dict = None) -> str:
    """Animated hub-and-spoke graph of the REAL topology — the Supervisor routes to Architect /
    Infra / Medic and each returns to it (NOT a linear pipeline). The node running NOW pulses with a
    marching-ants ring and the active edge animates in the travel direction; finished nodes are solid
    green, failed red, idle dark. Rendered via st.components.html (an iframe, so the CSS animations
    survive — st.markdown would sanitise them). Re-render ONLY on state change so it doesn't restart."""
    import math
    # (cx, cy, w, h)
    _POS = {
        "supervisor": (300, 56, 176, 66),
        "architect":  (104, 250, 150, 60),
        "infra":      (300, 250, 150, 60),
        "medic":      (496, 250, 150, 60),
    }
    _LBL = {"supervisor": "Supervisor", "architect": "Architect", "infra": "Infra", "medic": "Medic"}
    _CLS = {"active": "n-active", "completed": "n-done", "failed": "n-failed",
            "pending": "n-pending", "hub": "n-hub", "warn": "n-warn"}

    def _short(sx, sy, dx, dy, gap):
        L = math.hypot(dx - sx, dy - sy) or 1
        return dx - (dx - sx) / L * gap, dy - (dy - sy) / L * gap

    def _line(a, b):
        hot = last_edge is not None and {a, b} == set(last_edge)
        src, dst = (last_edge if hot else (a, b))
        sx, sy = _POS[src][:2]
        dx, dy = _POS[dst][:2]
        sx, sy = _short(dx, dy, sx, sy, 38)   # pull start off the src box
        dx, dy = _short(_POS[src][0], _POS[src][1], dx, dy, 38)  # and end off the dst box
        cls = "e-active" if hot else "e"
        mk = ' marker-end="url(#arrow)"' if hot else ''
        return f'<line x1="{sx:.0f}" y1="{sy:.0f}" x2="{dx:.0f}" y2="{dy:.0f}" class="{cls}"{mk}/>'

    _GLYPH = {"completed": ("✓", "#4ade80"), "failed": ("✗", "#f87171"),
              "warn": ("!", "#fbbf24")}

    def _box(key):
        cx, cy, w, h = _POS[key]
        status = node_statuses.get(key, "pending")
        cls = _CLS.get(status, "n-pending")
        parts = [
            f'<rect x="{cx-w/2:.0f}" y="{cy-h/2:.0f}" width="{w}" height="{h}" rx="13" class="{cls}"/>',
            f'<text x="{cx}" y="{cy+6}" text-anchor="middle" class="lbl">{_LBL[key]}</text>',
        ]
        # Status glyph at the top-right corner (n8n / React Flow convention).
        if status in _GLYPH:
            g, gc = _GLYPH[status]
            parts.append(f'<text x="{cx+w/2-13:.0f}" y="{cy-h/2+17:.0f}" '
                         f'class="glyph" fill="{gc}">{g}</text>')
        # Run-count badge at the top-left when a node ran more than once — visualises the
        # multi-agent looping (the Supervisor hub + the architect's multi-phase / self-heal runs).
        n = (run_counts or {}).get(key, 0)
        if n >= 2:
            parts.append(f'<text x="{cx-w/2+9:.0f}" y="{cy-h/2+17:.0f}" class="cnt">×{n}</text>')
        return "".join(parts)

    svg = (
        '<svg viewBox="0 0 600 312" width="100%" style="max-height:330px;display:block;margin:auto;" '
        'xmlns="http://www.w3.org/2000/svg">'
        '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="6" refY="3" '
        'orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#38bdf8"/></marker></defs>'
        + _line("supervisor", "architect") + _line("supervisor", "infra")
        + _line("supervisor", "medic")
        + _box("supervisor") + _box("architect") + _box("infra") + _box("medic")
        + '</svg>'
    )
    return (
        "<style>"
        "body{margin:0;background:transparent;}"
        "@keyframes march{to{stroke-dashoffset:-28;}}"
        "@keyframes pulse{0%,100%{stroke-width:2.5;}50%{stroke-width:5;}}"
        ".e{stroke:#2a3850;stroke-width:1.6;}"
        ".e-active{stroke:#38bdf8;stroke-width:3;stroke-dasharray:9 5;animation:march .55s linear infinite;}"
        "rect{fill:#0b1220;}"
        ".n-pending{stroke:#334155;stroke-width:1.5;}"
        # hub = the Supervisor while running: a steady, neutral coordinator look (NOT done-green
        # from being visited, NOT the pulsing active ring). Its green/red/amber comes only at the
        # terminal state, so it reflects the MISSION outcome.
        ".n-hub{fill:#0f1830;stroke:#64748b;stroke-width:2;}"
        ".n-done{fill:#04210f;stroke:#4ade80;stroke-width:2;}"
        ".n-failed{fill:#2a0f0f;stroke:#f87171;stroke-width:2.5;}"
        ".n-warn{fill:#2a210a;stroke:#fbbf24;stroke-width:2.5;}"
        ".n-active{fill:#0c2942;stroke:#38bdf8;stroke-dasharray:11 6;"
        "animation:march .7s linear infinite,pulse 1.3s ease-in-out infinite;"
        "filter:drop-shadow(0 0 7px rgba(56,189,248,.7));}"
        ".lbl{fill:#e2e8f0;font-family:Helvetica,Arial,sans-serif;font-size:17px;font-weight:600;}"
        ".glyph{font-family:Helvetica,Arial,sans-serif;font-size:16px;font-weight:700;}"
        ".cnt{fill:#94a3b8;font-family:Helvetica,Arial,sans-serif;font-size:12px;font-weight:600;}"
        "</style>" + svg
    )


_AGENT_TL_LABELS = {
    "supervisor": "🎯 Supervisor", "architect": "🏗️ Architect",
    "infra": "⚙️ Infra", "medic": "🏥 Medic", "execute_tools": "🔧 Tools",
}


def _structure_messages(messages) -> list:
    """Turn a node super-step's delta messages into ordered sub-steps for the timeline:
    tool CALLS (name + args), tool RESULTS (output), and TEXT (the agent's own message).
    The architect/infra run their tools in-node, so a single super-step carries the whole
    AIMessage(tool_calls)→ToolMessage(result) trace — exactly LangSmith's per-run detail.

    Each result is placed IMMEDIATELY AFTER the call it belongs to (matched by tool_call_id),
    even when one LLM message issues several tool_calls at once (where the raw order is
    call,call,…,result,result,…). Results whose call lives in an earlier super-step (the
    medic → execute_tools split) are emitted standalone, in order."""
    def _tc_field(tc, key, default):
        return tc.get(key, default) if isinstance(tc, dict) else getattr(tc, key, default)

    # Index this batch's results by their originating call id.
    results_by_id = {}
    for msg in messages:
        tid = getattr(msg, "tool_call_id", None)
        if tid is not None and not getattr(msg, "tool_calls", None):
            results_by_id[tid] = {"kind": "result", "name": getattr(msg, "name", "") or "",
                                  "text": str(getattr(msg, "content", "") or "")}

    steps, consumed = [], set()
    for msg in messages:
        tcs = getattr(msg, "tool_calls", None)
        if tcs:
            for tc in tcs:
                tid = _tc_field(tc, "id", None)
                call = {"kind": "call", "name": _tc_field(tc, "name", "tool"),
                        "args": _tc_field(tc, "args", {}) or {}}
                if tid in results_by_id:          # NEST the result INSIDE its call (one click = in+out)
                    call["result"] = results_by_id[tid]
                    consumed.add(tid)
                # Hide redundant no-op calls (e.g. a 2nd read_data_schema / a write the tool
                # skipped because it already ran) — the result starts with "Skipped:". Pure noise.
                _r = call.get("result")
                if _r and str(_r.get("text", "")).lstrip().startswith("Skipped:"):
                    continue
                steps.append(call)
            txt = (getattr(msg, "content", "") or "")
            if isinstance(txt, str) and txt.strip():
                steps.append({"kind": "text", "text": txt.strip()})
            continue
        tid = getattr(msg, "tool_call_id", None)
        if tid is not None:
            if tid not in consumed:               # call was in an earlier super-step
                steps.append(results_by_id.get(tid, {
                    "kind": "result", "name": getattr(msg, "name", "") or "",
                    "text": str(getattr(msg, "content", "") or "")}))
                consumed.add(tid)
            continue
        txt = getattr(msg, "content", "") or ""
        if isinstance(txt, str) and txt.strip():
            steps.append({"kind": "text", "text": txt.strip()})
        elif txt:  # non-str content (rare) — stringify
            steps.append({"kind": "text", "text": str(txt)})
    return steps


def _summarize_calls(names: list) -> str:
    """`['a','b','b']` → `a · b ×2`, order-preserving — the collapsed step summary."""
    seen = []
    counts = {}
    for n in names:
        if n not in counts:
            seen.append(n)
        counts[n] = counts.get(n, 0) + 1
    return " · ".join(n + (f" ×{counts[n]}" if counts[n] > 1 else "") for n in seen)


def _cell(txt, limit: int = 6000) -> str:
    """Truncate + collapse blank lines + escape, for safe embedding in the st.markdown HTML.
    CRITICAL: a BLANK line inside embedded HTML makes Streamlit's markdown renderer EXIT html
    mode and dump the rest of the timeline as literal <details>/<pre> tags (seen when a large
    query_vector_store result — a full standard with blank lines — is shown). Collapsing blank
    lines keeps the HTML one continuous block. Single newlines stay (pre-wrap shows them)."""
    import html as _html
    import re as _re
    s = str(txt)
    if len(s) > limit:
        s = s[:limit] + "\n…(truncated)"
    s = _re.sub(r"\n[ \t]*\n+", "\n", s)   # drop blank lines — the markdown-in-HTML killer
    return _html.escape(s)


def _scroll_to_top():
    """Scroll the main view back to the top. Used on wizard step ENTRY so the user lands on the
    section heading/summary, not wherever the previous step left the scroll (e.g. the deploy
    button at the bottom). Targets the Streamlit main scroll container across versions + the
    window itself; a tiny random query keeps the iframe unique so it re-fires on each entry."""
    import random
    components.html(
        f"""
        <script>
          const _t = {random.random()};
          const d = window.parent.document;
          const main = d.querySelector(
            'section.main, section[data-testid="stMain"], .stMain, [data-testid="stAppViewContainer"] > .main'
          );
          if (main) {{ main.scrollTo({{top: 0, behavior: 'auto'}}); }}
          window.parent.scrollTo({{top: 0, behavior: 'auto'}});
        </script>
        """,
        height=0,
    )


def _render_substep_html(s: dict) -> str:
    """One nested, expandable sub-step (LangSmith-style): a tool call shows its input args,
    a result shows the tool output, plain text renders inline."""
    import html as _html
    import json as _json
    pre = ("color:#cbd5e1;font-size:0.74rem;white-space:pre-wrap;margin:0.25rem 0 0;"
           "background:rgba(2,6,18,0.85);border-radius:5px;padding:0.45rem;"
           "overflow:auto;max-height:340px;")
    summ = ("cursor:pointer;font-size:0.78rem;font-weight:600;padding:0.3rem 0.45rem;"
            "border-radius:5px;")
    det = "margin:0.28rem 0 0.28rem 0.4rem;"

    if s["kind"] == "call":
        name = _html.escape(str(s.get("name", "tool")))
        args = s.get("args", {}) or {}
        rows = []
        for k, v in (args.items() if isinstance(args, dict) else []):
            vs = v if isinstance(v, str) else _json.dumps(v, default=str, ensure_ascii=False)
            rows.append(f"<b style='color:#7dd3fc;'>{_html.escape(str(k))}</b>: "
                        f"{_cell(vs, 4000)}")
        body = "<br>".join(rows) if rows else "<span style='color:#475569;'>(no input)</span>"
        # Result nested INSIDE the call expander: one click shows BOTH the input args and the
        # tool output. A call whose result is not in this super-step (e.g. a cross-node split)
        # shows a note instead of silently dropping the tool.
        res = s.get("result")
        if res is not None:
            result_html = (
                '<div style="margin:0.5rem 0 0.1rem;color:#4ade80;font-size:0.74rem;'
                'font-weight:600;">↳ result</div>'
                f'<pre style="{pre}">{_cell(res.get("text", ""))}</pre>')
        else:
            result_html = (
                '<div style="margin-top:0.5rem;color:#64748b;font-size:0.72rem;">'
                '↳ result not captured in this step — see the full trace in LangSmith</div>')
        return (f'<details style="{det}"><summary style="{summ}color:#38bdf8;">'
                f'🔧 <span style="color:#e2e8f0;">{name}</span></summary>'
                f'<div style="{pre}">{body}</div>{result_html}</details>')

    if s["kind"] == "result":
        name = _html.escape(str(s.get("name", "")))
        head = f' <span style="color:#94a3b8;">{name}</span>' if name else ""
        return (f'<details style="{det}"><summary style="{summ}color:#4ade80;">'
                f'↳ result{head}</summary>'
                f'<pre style="{pre}">{_cell(s.get("text", ""))}</pre></details>')

    return (f'<div style="margin:0.28rem 0 0.28rem 0.4rem;font-size:0.78rem;color:#cbd5e1;">'
            f'💬 <span style="color:#94a3b8;">agent</span>'
            f'<pre style="{pre}">{_cell(s.get("text", ""))}</pre></div>')


def _render_agent_log_html(agent_events: list) -> str:
    """LangSmith-style execution timeline — one collapsible entry per node super-step, in
    order. Collapsed shows the agent + a summary of the tools it called; expanded reveals the
    nested tool calls (with input args) and their outputs. The newest step is open by default.
    Keeps EVERY step (architect Discovery→Schema→Implementation→self-heal patches, medic
    diagnosis + tools), not just the last. Fed by the `agent_event` updates from _run_agent."""
    import html as _html
    if not agent_events:
        return ('<p style="color:#334155;font-size:0.82rem;padding:0.6rem;">'
                'Waiting for the first agent step…</p>')

    summary_style = (
        "cursor:pointer;color:#e2e8f0;font-size:0.85rem;font-weight:600;"
        "padding:0.5rem 0.6rem;border-radius:7px;background:rgba(15,23,42,0.7);"
    )
    det_style = ("margin:0.35rem 0;border-left:2px solid #1e293b;padding:0 0 0.2rem 0.55rem;")

    blocks = []
    last = len(agent_events) - 1
    for i, (agent, steps) in enumerate(agent_events):
        label = _AGENT_TL_LABELS.get(agent, agent.title())
        calls = [s["name"] for s in steps if s.get("kind") == "call"]
        if calls:
            summ = _summarize_calls(calls)
        else:
            txt = next((s.get("text", "") for s in steps if s.get("kind") == "text"), "")
            txt = " ".join(txt.split())
            summ = (txt[:90] + "…") if len(txt) > 90 else (txt or "—")
        inner = "".join(_render_substep_html(s) for s in steps)
        open_attr = " open" if i == last else ""
        blocks.append(
            f'<details{open_attr} style="{det_style}">'
            f'<summary style="{summary_style}">'
            f'<span style="color:#475569;">#{i + 1}</span> {label} '
            f'<span style="color:#64748b;font-weight:400;">— {_html.escape(summ)}</span>'
            f'</summary>{inner}</details>'
        )
    return "".join(blocks)


# ---------------------------------------------------------------------------
# Agent runner (background thread)
# ---------------------------------------------------------------------------

def _run_agent(pipe_conf, db_conf, rules_conf, infra_conf,
               pipeline_id, task, log_q: queue.Queue, state_q: queue.Queue):

    class _QH(logging.Handler):
        def emit(self, record):
            log_q.put(("log", f"  {record.name} — {record.getMessage()}"))

    handler = _QH()
    root = logging.getLogger()
    root.addHandler(handler)

    try:
        # MissionFailedError/mission_failure_summary live in main.py; import here (lazy,
        # as before) and early so the except handler below always has the name bound.
        from main import MissionFailedError, mission_failure_summary
        graph = get_graph()
        from agents.state import build_initial_state  # lazy, matches get_graph's deferred import
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        project_id = f"{pipeline_id.upper()}-{timestamp}"
        os.environ["PROJECT_ID"] = project_id
        # Mirror main.py: agent-runtime cloud_get() resolves creds per CLOUD_PROVIDER (aws), and the
        # file tools read the runtime from PIPELINE_PLATFORM (kubernetes on EKS).
        os.environ["CLOUD_PROVIDER"] = pipe_conf.get("cloud_provider", "aws")
        os.environ["PIPELINE_PLATFORM"] = infra_conf.get("provider", "kubernetes").lower()
        # Same PII posture as the CLI: mask sample rows to the LLM + enforce the anonymization gate.
        os.environ["PII_SENSITIVE"] = "true" if pipe_conf.get("pii_sensitive") else "false"

        # Shared factory with main.py — the two entry points can never drift on which state keys are
        # set. (Structured JSON logging via setup_logging is CLI-only by design; here the live run
        # streams to the UI through the _QH queue handler installed in _run_agent, so we must NOT call
        # setup_logging — it would drop that handler.)
        initial_state = build_initial_state(
            project_id=project_id,
            task=task,
            raw_configs={"pipeline": pipe_conf, "database": db_conf,
                         "rules": rules_conf, "infrastructure": infra_conf},
            target_infra=infra_conf.get("service_name", pipe_conf.get("cloud_provider", "unknown")),
        )

        log_q.put(("head", f"🚀  Pipeline : {pipeline_id}"))
        log_q.put(("head", f"🆔  Project  : {project_id}"))
        log_q.put(("head", f"☁️   Cloud    : {pipe_conf.get('cloud_provider','?').upper()}"))
        log_q.put(("sep",  "─" * 55))
        state_q.put({"project_id": project_id})

        mission_status = ""
        _graph_stream = graph.stream(initial_state, config={
            "run_name": f"streamlit_{pipeline_id}_{timestamp}",
            "recursion_limit": 200,
            "configurable": {"thread_id": project_id},
        })
        for output in _graph_stream:
            for node_name, state_update in output.items():
                log_q.put(("node", f"\n{_NODE_LABELS.get(node_name, node_name.upper())}"))
                if state_update.get("mission_status"):
                    mission_status = state_update["mission_status"]
                # Fail-closed into the trace: a FINISH without a verified mission is thrown
                # INTO the stream (the proven main.py:_consume_stream path) so the LangSmith
                # ROOT run is recorded as errored — a green trace must mean a verified
                # mission, never just "the graph reached FINISH".
                if state_update.get("next_step") == "FINISH" and mission_status != "verified":
                    _graph_stream.throw(MissionFailedError(
                        f"mission_status='{mission_status or 'unset'}' — "
                        f"{mission_failure_summary(mission_status)}"))
                if nxt := state_update.get("next_step"):
                    if nxt != "FINISH":
                        log_q.put(("route", f"  → {nxt.upper()}"))
                if state_update.get("architect_status") == "completed":
                    log_q.put(("ok", "  ✅ Architect phase complete"))
                if state_update.get("infra_status") == "completed":
                    log_q.put(("ok", "  ✅ Infra phase complete"))
                if state_update.get("github_done"):
                    sha = state_update.get("last_push_sha", "")
                    log_q.put(("ok", f"  ✅ Pushed to GitHub{' — SHA ' + sha[:7] if sha else ''}"))
                if files := state_update.get("written_files"):
                    state_q.put({"written_files": files})

                # ── Pipeline visualization updates ────────────────────────
                if node_name in ("supervisor", "architect", "infra", "medic"):
                    nxt = state_update.get("next_step", "").lower()
                    state_q.put({"node_update": {
                        "completed": node_name,
                        "next_active": nxt if nxt and nxt != "finish" else None,
                    }})

                # Timeline capture — structured sub-steps (tool calls + args + results + text)
                # per super-step, in execution order. execute_tools carries the medic's tool
                # results. One appended entry per step (the timeline mirrors the ping-pong).
                if node_name in ("supervisor", "architect", "infra", "medic", "execute_tools"):
                    steps = _structure_messages(state_update.get("messages", []))
                    if steps:
                        state_q.put({"agent_event": (node_name, steps)})

                # Healing cycle: medic reset architect or infra status to pending
                if node_name == "medic" and (
                    state_update.get("architect_status") == "pending"
                    or state_update.get("infra_status") == "pending"
                ):
                    state_q.put({"healing_cycle": 1})

        log_q.put(("sep", "─" * 55))
        if mission_status == "verified":
            log_q.put(("ok",  "🎉  Deployment complete — verified end-to-end!"))
            state_q.put({"status": "DONE", "mission_status": mission_status})
        else:
            # The graph ENDED but without verified deployment (fix loop abandoned,
            # CI unverified, or an LLM-fallback FINISH) — never celebrate a failure.
            from main import mission_failure_summary
            reason = mission_failure_summary(mission_status)
            log_q.put(("err", f"❌  Mission failed (mission_status={mission_status or 'unset'}): {reason}"))
            state_q.put({"status": "ERROR", "error": f"Unverified termination: {reason}",
                         "mission_status": mission_status})

    except MissionFailedError:
        # Thrown above on a non-verified FINISH → the LangSmith root run is now errored.
        # Report ❌ and PRESERVE mission_status so the Supervisor node still colours amber
        # (escalated / ci_unverified) vs red — identical UI verdict as before, now with a
        # matching red trace instead of a misleading green one.
        reason = mission_failure_summary(mission_status)
        log_q.put(("err", f"❌  Mission failed (mission_status={mission_status or 'unset'}): {reason}"))
        state_q.put({"status": "ERROR", "error": f"Unverified termination: {reason}",
                     "mission_status": mission_status})
    except Exception as exc:
        log_q.put(("err", f"❌  {exc}"))
        state_q.put({"status": "ERROR", "error": str(exc)})
    finally:
        root.removeHandler(handler)


def _start_run(pipe_conf, db_conf, rules_conf, infra_conf, pipeline_id, task):
    lq, sq = queue.Queue(), queue.Queue()
    st.session_state.update({
        "log_q": lq, "state_q": sq,
        "pipeline_meta": {
            "pipeline_id": pipeline_id,
            "cloud": pipe_conf.get("cloud_provider", "?").upper(),
            "db":    db_conf.get("db_type", "?"),
            "rules": len(rules_conf.get("quality_standards", [])),
        },
        "written_files": [], "run_status": "running",
        "node_statuses":  {"supervisor": "pending", "architect": "pending",
                            "infra": "pending", "medic": "pending"},
        "agent_events": [],
        "run_counts": {},
        "healing_cycles": 0,
        "run_start_time": time.time(),
    })
    t = threading.Thread(
        target=_run_agent,
        args=(pipe_conf, db_conf, rules_conf, infra_conf, pipeline_id, task, lq, sq),
        daemon=True,
    )
    t.start()
    st.session_state.agent_thread = t


# Minimum bootstrap-output keys a cloud needs before an NL-authored pipeline can deploy.
# Without these, _build_from_answers fills REQUIRES_BOOTSTRAP_OUTPUTS sentinels and the run
# fails deep inside terraform with a cryptic error. We surface a clear message up front.
_BOOTSTRAP_REQUIRED_KEYS = {
    "aws":   ("state_bucket", "aws_account_id", "pipeline_irsa_role_name"),
}


def _bootstrap_ready(cloud: str) -> tuple[bool, str]:
    """True when .bootstrap_outputs.json carries what an NL deploy needs for `cloud`."""
    try:
        from utils.nlp_parser import _load_bootstrap_outputs
        outputs = _load_bootstrap_outputs().get(cloud, {})
    except Exception as e:
        return False, f"Could not read .bootstrap_outputs.json: {e}"
    missing = [k for k in _BOOTSTRAP_REQUIRED_KEYS.get(cloud, ()) if not outputs.get(k)]
    if missing:
        return False, (
            f"The **{cloud.upper()}** baseline is not wired for a runtime deploy "
            f"(missing bootstrap outputs: {', '.join(missing)}).\n\n"
            f"Run the one-time bridge after bootstrapping:\n"
            f"`python scripts/export_bootstrap_outputs.py {cloud}`\n\n"
            f"(or `make bootstrap-{cloud}`, which now exports automatically)."
        )
    return True, ""


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------

st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(29,78,216,0.18) 0%, rgba(14,165,233,0.12) 50%, rgba(99,102,241,0.18) 100%);
    border: 1px solid rgba(56,189,248,0.3);
    border-radius: 18px;
    padding: 2rem 2.5rem 1.8rem;
    margin-bottom: 1.8rem;
    backdrop-filter: blur(16px);
    position: relative;
    overflow: hidden;
">
  <!-- faint grid lines for depth -->
  <div style="
    position:absolute; inset:0; opacity:0.04;
    background-image: linear-gradient(rgba(255,255,255,.5) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(255,255,255,.5) 1px, transparent 1px);
    background-size: 32px 32px;
    border-radius: 18px;
  "></div>

  <h1 style="
    margin: 0 0 0.5rem;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 60%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
  ">🤖 Self-Healing Big Data Pipeline Agent</h1>

  <p style="margin:0; color:#94a3b8; font-size:1rem; line-height:1.6;">
    Describe a pipeline in plain English — the agents design, deploy, and monitor it on&nbsp;
    <span style="color:#f97316; font-weight:600;">AWS</span>:
    PostgreSQL → cleaned Parquet on S3 (partitioned by run_date) → Trino → Grafana.
  </p>

  <div style="display:flex; gap:1.5rem; margin-top:1.2rem; flex-wrap:wrap;">
    <span style="
      background:rgba(249,115,22,0.15); color:#f97316;
      border:1px solid rgba(249,115,22,0.3); border-radius:8px;
      padding:0.25rem 0.8rem; font-size:0.8rem; font-weight:600; letter-spacing:0.05em;
    ">🟠 AWS EKS + S3</span>
    <span style="
      background:rgba(56,189,248,0.12); color:#38bdf8;
      border:1px solid rgba(56,189,248,0.3); border-radius:8px;
      padding:0.25rem 0.8rem; font-size:0.8rem; font-weight:600; letter-spacing:0.05em;
    ">🗂️ Parquet + run_date partitions</span>
    <span style="
      background:rgba(129,140,248,0.12); color:#818cf8;
      border:1px solid rgba(129,140,248,0.3); border-radius:8px;
      padding:0.25rem 0.8rem; font-size:0.8rem; font-weight:600; letter-spacing:0.05em;
    ">⚡ Self-Healing</span>
    <span style="
      background:rgba(192,132,252,0.12); color:#c084fc;
      border:1px solid rgba(192,132,252,0.3); border-radius:8px;
      padding:0.25rem 0.8rem; font-size:0.8rem; font-weight:600; letter-spacing:0.05em;
    ">🔮 Trino SQL</span>
    <span style="
      background:rgba(251,191,36,0.12); color:#fbbf24;
      border:1px solid rgba(251,191,36,0.3); border-radius:8px;
      padding:0.25rem 0.8rem; font-size:0.8rem; font-weight:600; letter-spacing:0.05em;
    ">📊 Grafana Dashboards</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    # ── Header ──────────────────────────────────────────────
    st.markdown(
        '<p class="sb-title" style="margin:0 0 0.15rem;">Environment</p>'
        '<p style="margin:0 0 1rem; font-size:1.05rem; font-weight:700; color:#e2e8f0;">Status</p>',
        unsafe_allow_html=True,
    )

    # ── API Keys (read-only status) ──────────────────────────
    checks = {
        ("Gemini" if _using_gemini() else "OpenAI"): bool(_nl_api_key()),
        "Pinecone": bool(os.getenv("PINECONE_API_KEY")),
        "GitHub":   bool(os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")),
    }

    key_rows = "".join(
        f'<div class="sb-row">'
        f'<span class="sb-lbl">🔑 {name}</span>'
        f'<span class="{"sb-ok" if ok else "sb-err"}">{"● set" if ok else "○ missing"}</span>'
        f'</div>'
        for name, ok in checks.items()
    )
    st.markdown(
        f'<div class="sb-card"><p class="sb-title">API Keys</p>{key_rows}</div>',
        unsafe_allow_html=True,
    )

    # ── AWS baseline status (read-only; NL-deploy readiness) ─────
    # The baseline (EKS + ECR + state backend + RDS + shared IRSA role) is provisioned ONCE via
    # `make bootstrap-aws`. An NL deploy needs it ready — the same check the deploy guard uses
    # (_bootstrap_ready). "Not provisioned" is shown neutrally: it's a one-time step.
    _ready, _ = _bootstrap_ready("aws")
    _status = ('<span class="sb-ok">● ready</span>' if _ready
               else '<span style="color:#64748b;">○ not provisioned</span>')
    st.markdown(
        f'<div class="sb-card"><p class="sb-title">AWS Baseline</p>'
        f'<div class="sb-row"><span class="sb-lbl">🟠 AWS</span>{_status}</div></div>',
        unsafe_allow_html=True,
    )
    if not _ready:
        st.markdown(
            '<p style="color:#475569;font-size:0.72rem;margin:0.3rem 0 0;line-height:1.45;">'
            'One-time: <code style="color:#7dd3fc;">make bootstrap-aws</code> '
            '(provisions the baseline + auto-exports outputs).</p>',
            unsafe_allow_html=True,
        )

    # ── Footer ───────────────────────────────────────────────
    st.markdown(
        '<p style="color:#1e293b; font-size:0.72rem; text-align:center; margin-top:1rem;">'
        'LangGraph · OpenAI · Pinecone</p>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# ARCHITECTURE REPORT RENDERER
# ---------------------------------------------------------------------------

def _render_arch_report(report: dict):
    specs = report.get("specs", {})
    opt   = report.get("option") or {}
    color = "#f97316"

    # ── Summary strip ─────────────────────────────────────────
    compliance_str = ", ".join(specs.get("compliance", [])) or "None detected"
    ml_badge = ' &nbsp;·&nbsp; 🤖 ML pipeline' if specs.get("needs_ml") else ""
    etl_badge = ' &nbsp;·&nbsp; ⚙️ Heavy transforms' if specs.get("heavy_transforms") else ""
    st.markdown(
        f'<div style="background:rgba(29,78,216,0.1);border:1px solid rgba(56,189,248,0.2);'
        f'border-radius:12px;padding:0.8rem 1.1rem;margin:0.8rem 0;">'
        f'<p style="margin:0;font-size:0.72rem;color:#475569;text-transform:uppercase;'
        f'letter-spacing:0.08em;">Architecture Report</p>'
        f'<p style="margin:0.2rem 0 0.6rem;color:#e2e8f0;font-weight:600;">{report.get("summary", "")}</p>'
        f'<span style="color:#64748b;font-size:0.8rem;">'
        f'📦 ~{specs.get("data_volume_gb_day",50)} GB/day &nbsp;·&nbsp; '
        f'🔄 {specs.get("frequency","daily")} &nbsp;·&nbsp; '
        f'🌍 {specs.get("region","eu-central-1")} &nbsp;·&nbsp; '
        f'🔒 {compliance_str}{ml_badge}{etl_badge}'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    # ── AWS stack card ────────────────────────────────────────
    pros_html  = "".join(f'<p class="arch-pro">✓ {p}</p>' for p in opt.get("pros", []))
    cons_html  = "".join(f'<p class="arch-con">✗ {p}</p>' for p in opt.get("cons", []))
    items_html = "".join(
        f'<div style="display:flex;justify-content:space-between;font-size:0.71rem;'
        f'padding:0.12rem 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
        f'<span style="color:#475569;">{k}</span>'
        f'<span style="color:#64748b;">${v:.2f}</span></div>'
        for k, v in opt.get("items", {}).items()
    )
    st.markdown(
        f'<div class="arch-card arch-card-rec">'
        f'<p class="arch-cloud-name" style="color:{color};">🟠 AWS — S3 data lake + Trino on EKS</p>'
        f'<p class="arch-price" style="color:{color};">'
        f'${opt.get("total", 0):.0f}<span style="font-size:0.8rem;color:#64748b;font-weight:400;">/mo</span></p>'
        f'<p class="arch-label">{opt.get("label", "")}</p>'
        f'{pros_html}{cons_html}'
        f'<details style="margin-top:0.6rem;">'
        f'<summary style="color:#475569;font-size:0.74rem;cursor:pointer;">Cost breakdown</summary>'
        f'<div style="margin-top:0.4rem;">{items_html}</div>'
        f'</details></div>',
        unsafe_allow_html=True,
    )

    if report.get("fit_reason"):
        st.markdown(
            f'<div style="background:rgba(74,222,128,0.07);border:1px solid rgba(74,222,128,0.2);'
            f'border-radius:10px;padding:0.7rem 1rem;margin-top:0.8rem;">'
            f'<span style="font-weight:700;color:#4ade80;">💡 How this workload fits</span>'
            f'<p style="margin:0.3rem 0 0;color:#94a3b8;font-size:0.85rem;">'
            f'{report["fit_reason"]}</p></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# RULES SELECTOR HELPER
# ---------------------------------------------------------------------------

def _render_rules_selector(key_prefix: str):
    from utils.rules_loader import (
        load_demo_rules, list_demo_domains, parse_rules_file,
        SIMPLE_TEMPLATE,
    )

    mode = st.radio(
        "Rules source",
        ["📋 Demo rules", "📝 Extract from description", "📁 Upload file"],
        key=f"rules_mode_{key_prefix}",
        horizontal=True,
        label_visibility="collapsed",
    )

    rules_conf = None

    if mode == "📋 Demo rules":
        domains = list_demo_domains()
        domain  = st.selectbox("Domain", domains, key=f"rules_domain_{key_prefix}")
        rules_conf = load_demo_rules(domain)

    elif mode == "📝 Extract from description":
        st.markdown(
            '<p style="color:#64748b;font-size:0.82rem;margin:0.3rem 0;">'
            'Rules will be extracted automatically when you click <b>Analyze &amp; Plan</b>.</p>',
            unsafe_allow_html=True,
        )
        rules_conf = st.session_state.get(f"rules_conf_{key_prefix}")

    elif mode == "📁 Upload file":
        st.download_button(
            "⬇️ Download template",
            data=SIMPLE_TEMPLATE,
            file_name="rules_template.yaml",
            mime="text/yaml",
            key=f"rules_dl_{key_prefix}",
        )
        uploaded = st.file_uploader(
            "Upload YAML or JSON",
            type=["yaml", "yml", "json"],
            key=f"rules_upload_{key_prefix}",
            label_visibility="collapsed",
        )
        if uploaded:
            try:
                rules_conf = parse_rules_file(uploaded.read(), uploaded.name)
                st.session_state[f"rules_conf_{key_prefix}"] = rules_conf
            except ValueError as e:
                st.error(str(e))
                rules_conf = None

    # Preview
    if rules_conf:
        standards = rules_conf.get("quality_standards", [])
        domain_lbl = rules_conf.get("domain", "")
        st.markdown(
            f'<p style="color:#4ade80;font-size:0.8rem;margin:0.5rem 0 0.3rem;">'
            f'✓ {len(standards)} rule{"s" if len(standards)!=1 else ""} loaded'
            f'{" · domain: " + domain_lbl if domain_lbl else ""}</p>',
            unsafe_allow_html=True,
        )
        with st.expander("Preview rules", expanded=False):
            for s in standards:
                st.markdown(
                    f'<div style="border-left:2px solid rgba(56,189,248,0.3);'
                    f'padding:0.3rem 0.7rem;margin:0.3rem 0;">'
                    f'<p style="margin:0;color:#e2e8f0;font-size:0.82rem;font-weight:600;">'
                    f'{s.get("capability","")}</p>'
                    f'<p style="margin:0;color:#64748b;font-size:0.76rem;">'
                    f'{s.get("logic","")}</p>'
                    f'<span style="color:#f97316;font-size:0.72rem;">'
                    f'on failure → {s.get("on_failure_action","")}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.session_state[f"rules_conf_{key_prefix}"] = rules_conf


def _get_active_rules(key_prefix: str) -> dict | None:
    """Return the rules_conf chosen for this tab, or None to use pipeline defaults."""
    return st.session_state.get(f"rules_conf_{key_prefix}")


# ---------------------------------------------------------------------------
# COST CHARTS HELPER
# ---------------------------------------------------------------------------

def _render_cost_items(est: dict):
    """Itemised AWS cost table + total."""
    _df = pd.DataFrame([{"Service": k, "$/month": round(v, 2)} for k, v in est["items"].items()])
    st.dataframe(
        _df, use_container_width=True, hide_index=True,
        column_config={"$/month": st.column_config.NumberColumn("$/month", format="$%.2f")},
    )
    st.markdown(
        f'<p style="color:#94a3b8;font-size:0.85rem;margin-top:0.3rem;">'
        f'Total: <b style="color:#f97316;">${est["total"]:.0f}/mo</b></p>',
        unsafe_allow_html=True,
    )


def _render_cost_charts(size_gb: int = 50, key: str = "cost"):
    # `key` is kept for call-site compatibility (several tabs render this helper).
    from utils.cost_estimator import estimate_monthly_cost
    est = estimate_monthly_cost("aws", storage_gb=size_gb)

    bar_df = pd.DataFrame({"Service": list(est["items"]), "$/month": list(est["items"].values())}
                          ).set_index("Service")
    st.bar_chart(bar_df, color="#f97316", height=210, use_container_width=True)
    with st.expander("AWS cost breakdown", expanded=False):
        _render_cost_items(est)

    st.markdown(
        '<p style="color:#475569;font-size:0.78rem;margin:0.6rem 0 0;line-height:1.5;">'
        f'💡 Monthly <b>baseline</b> (EKS + RDS + ECR, provisioned <b>once</b>) plus this '
        f'pipeline\'s <b>data storage</b> ({size_gb} GB). A second pipeline adds only its S3 '
        'storage, not another cluster. <code>make aws-pause</code> stops RDS + EKS nodes between '
        'demo sessions.</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# COST ESTIMATION HELPER
# ---------------------------------------------------------------------------

def _render_cost_panel(selected_cloud: str | None = None):
    from utils.cost_estimator import PRICES_LAST_UPDATED, estimate_monthly_cost
    est = estimate_monthly_cost("aws")
    with st.expander("💰 Estimated monthly AWS infrastructure cost", expanded=False):
        _render_cost_items(est)
        st.markdown(
            '<p style="color:#334155;font-size:0.72rem;margin-top:0.6rem;">'
            f'* AWS list prices (verified {PRICES_LAST_UPDATED}) · eu-central-1 · fixed bootstrap '
            'footprint · 50 GB storage. Use the AWS Pricing Calculator for real estimates.</p>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# EXECUTION PLAN HELPER — illustrative "what will be created" (no run)
# ---------------------------------------------------------------------------

def _render_execution_plan(cloud: str, pipeline_id: str):
    """Illustrative pre-deploy plan: the deterministic AWS deliverable set. No agent run."""
    groups = [
        ("Pipeline code (ETL)", [
            f"scripts/{pipeline_id}.py — extract from RDS → business rules → Parquet (snappy) "
            "→ S3, partitioned by run_date",
            "requirements.txt — pandas, pyarrow, s3fs, boto3, sqlalchemy, trino, prometheus-client",
        ]),
        ("Catalog & observability", [
            "sql/setup_trino.sql — Trino external table over S3, run_date-partitioned",
            "dashboards/monitoring_specs.json — Grafana dashboard (rows processed / rejected, freshness)",
        ]),
        ("Kubernetes (EKS)", [
            "k8s/ — namespaces · configmaps · trino · grafana · prometheus(+pushgateway) · job",
            "Dockerfile — pipeline image",
        ]),
        ("Infrastructure (Terraform)", [
            "terraform/ — providers · main · variables · outputs · tfvars",
            "provisions: S3 bucket + IAM role (IRSA) + Glue catalog access",
        ]),
        ("CI/CD", [
            f".github/workflows/{pipeline_id}_pipeline.yml — build → ECR → deploy to EKS",
        ]),
    ]
    blocks = ""
    for title, items in groups:
        lis = "".join(f'<li style="margin:0.12rem 0;color:#94a3b8;">{it}</li>' for it in items)
        blocks += (
            f'<p style="margin:0.55rem 0 0.1rem;color:#7dd3fc;font-size:0.72rem;'
            f'text-transform:uppercase;letter-spacing:0.06em;">{title}</p>'
            f'<ul style="margin:0;padding-left:1.1rem;font-size:0.8rem;">{lis}</ul>'
        )
    st.markdown(
        f'<div style="background:rgba(10,18,40,0.6);border:1px solid rgba(56,189,248,0.18);'
        f'border-radius:12px;padding:0.85rem 1.1rem;margin:0.2rem 0 0.7rem;">{blocks}'
        f'<p style="margin:0.55rem 0 0;color:#334155;font-size:0.7rem;">'
        f'Illustrative — the agents generate the exact contents at deploy time. '
        f'Self-healing (Medic) repairs any CI/CD failure automatically.</p></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------------------------

tab_nl, tab_upload, tab_existing, tab_trino, tab_obs = st.tabs([
    "💬 Natural Language",
    "📁 Upload Dataset",
    "📂 Sample Pipelines",
    "🔮 Trino SQL",
    "📊 Observability",
])

# ── Upload Dataset tab ──────────────────────────────────────────────────────
with tab_upload:
    st.markdown(
        '<p style="color:#94a3b8;margin-bottom:0.8rem;">'
        'Upload a sample of your dataset — the agent detects the schema, '
        'spots PII fields, and pre-fills the business rules for you.</p>',
        unsafe_allow_html=True,
    )

    uploaded_ds = st.file_uploader(
        "Dataset",
        type=["csv", "json", "jsonl", "parquet"],
        label_visibility="collapsed",
        key="dataset_upload",
    )

    if uploaded_ds:
        with st.spinner("Analysing dataset…"):
            try:
                from utils.dataset_analyzer import analyze as analyze_dataset
                ds = analyze_dataset(uploaded_ds.read(), uploaded_ds.name)
                st.session_state.dataset_analysis = ds
            except Exception as e:
                st.error(str(e))
                ds = None

        if ds:
            stats = ds["stats"]
            # ── Stats row ──────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows",       f"{stats['rows']:,}")
            c2.metric("Columns",    stats["columns"])
            c3.metric("Size",       f"{stats['size_mb']} MB")
            c4.metric("Duplicates", stats["dup_rows"])

            # ── PII alert ──────────────────────────────────────
            if ds["pii_fields"]:
                pii_list = ", ".join(f"`{f}`" for f in ds["pii_fields"])
                st.markdown(
                    f'<div style="background:rgba(249,115,22,0.08);border:1px solid '
                    f'rgba(249,115,22,0.3);border-radius:10px;padding:0.65rem 1rem;margin:0.6rem 0;">'
                    f'<p style="margin:0;color:#f97316;font-weight:600;font-size:0.85rem;">'
                    f'⚠️ PII detected — {len(ds["pii_fields"])} field(s): {pii_list}</p>'
                    f'<p style="margin:0.2rem 0 0;color:#64748b;font-size:0.78rem;">'
                    f'Masking rules have been added automatically.</p></div>',
                    unsafe_allow_html=True,
                )

            # ── Quality issues ─────────────────────────────────
            if ds["quality_issues"]:
                with st.expander(f"⚠️ {len(ds['quality_issues'])} data quality issue(s) found"):
                    for iss in ds["quality_issues"]:
                        sev_color = "#f87171" if iss["severity"] == "high" else "#fbbf24"
                        st.markdown(
                            f'<div style="display:flex;gap:0.6rem;align-items:flex-start;'
                            f'padding:0.3rem 0;border-bottom:1px solid rgba(255,255,255,0.05);">'
                            f'<span style="color:{sev_color};font-size:0.8rem;min-width:60px;">'
                            f'{iss["severity"].upper()}</span>'
                            f'<span style="color:#94a3b8;font-size:0.8rem;">'
                            f'<b style="color:#e2e8f0;">{iss["column"]}</b> — {iss["detail"]}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            # ── Schema table ───────────────────────────────────
            with st.expander("📋 Schema", expanded=True):
                schema_df = pd.DataFrame([
                    {
                        "Column":   s["column"],
                        "Type":     s["type"],
                        "Nullable": "✓" if s["nullable"] else "✗",
                        "PII":      "⚠️" if s["column"] in ds["pii_fields"] else "",
                        "Sample values": " · ".join(s["sample"]),
                    }
                    for s in ds["schema"]
                ])
                st.dataframe(schema_df, use_container_width=True, hide_index=True)

            # ── Suggested rules summary ────────────────────────
            rules = ds["suggested_rules"]
            n     = len(rules["quality_standards"])
            st.markdown(
                f'<div style="background:rgba(74,222,128,0.07);border:1px solid '
                f'rgba(74,222,128,0.2);border-radius:10px;padding:0.65rem 1rem;margin-top:0.6rem;">'
                f'<p style="margin:0;color:#4ade80;font-weight:600;font-size:0.85rem;">'
                f'✓ {n} business rule{"s" if n!=1 else ""} auto-generated from your data</p>'
                f'<p style="margin:0.2rem 0 0;color:#64748b;font-size:0.78rem;">'
                f'Go to the <b>💬 Natural Language</b> tab, open ⚙️ Business Rules '
                f'and select <em>Extract from description</em> — or the rules are already '
                f'applied if you use the dataset analysis.</p></div>',
                unsafe_allow_html=True,
            )
            # Store rules so the NL tab can pick them up
            st.session_state["rules_conf_nl"] = rules
            st.session_state["rules_conf_ex"] = rules

            # ── Make this sample the pipeline's real source ────
            # The pipeline extracts from a DB table, not the uploaded file. To deploy a
            # brand-new pipeline on YOUR data, load this sample into the source database as
            # raw_<slug>; then in the wizard set source table = raw_<slug>.
            with st.expander("🔌 Make this your pipeline's source (load into the source DB)", expanded=False):
                _seed_slug = st.text_input("Pipeline slug", placeholder="e.g. my_sales",
                                           key="upload_seed_slug")
                if st.button("⬆️ Load sample into source DB", key="upload_seed_btn",
                             type="primary", disabled=not _seed_slug.strip()):
                    with st.spinner("Loading sample into the source database…"):
                        try:
                            from scripts.seed_chaos import seed_dataframe_to_source
                            uploaded_ds.seek(0)
                            _nm = uploaded_ds.name.lower()
                            if _nm.endswith(".parquet"):
                                _df = pd.read_parquet(uploaded_ds)
                            elif _nm.endswith(".jsonl"):
                                _df = pd.read_json(uploaded_ds, lines=True)
                            elif _nm.endswith(".json"):
                                _df = pd.read_json(uploaded_ds)
                            else:
                                _df = pd.read_csv(uploaded_ds)
                            _tbl = seed_dataframe_to_source(_df, _seed_slug.strip(), "aws")
                            st.session_state["nl_seed_table"] = _tbl
                            st.success(
                                f"✓ Seeded `{_tbl}` ({len(_df)} rows) into the AWS RDS source. "
                                f"In the wizard use **source table `{_tbl}`**, DB **postgres**."
                            )
                        except Exception as _se:
                            st.error(f"Could not seed the source DB: {_se}")

            # ── Cost chart preview ─────────────────────────────
            st.markdown("##### 💰 Estimated monthly cost based on your data volume")
            _render_cost_charts(size_gb=max(round(ds["size_gb_day"] * 30), 1), key="upload")

    else:
        st.markdown(
            '<div style="border:2px dashed rgba(56,189,248,0.2);border-radius:14px;'
            'padding:2.5rem;text-align:center;margin-top:0.5rem;">'
            '<p style="color:#334155;font-size:0.9rem;margin:0;">CSV · JSON · Parquet</p>'
            '<p style="color:#1e293b;font-size:0.78rem;margin:0.4rem 0 0;">'
            'Max ~100 MB recommended for preview</p>'
            '</div>',
            unsafe_allow_html=True,
        )

with tab_nl:
    # ── Session-state defaults ────────────────────────────────────────────
    _NL_DEFAULTS = {
        "nl_step": 0,
        "nl_description": "",
        "nl_intent": {},
        "nl_answers": {},
        "nl_rules": [],
        "nl_rules_mode": "detected",
        "nl_summary_ok": False,
        "nl_back_to_step": None,
        "nl_rules_initialized": False,
        # Rules sub-step view: "picker" (choose source) → "editor" (review/edit). One "← Back"
        # walks editor→picker, then picker→Fields. None = auto-decide on entry.
        "nl_rules_view": None,
        "nl_wizard_editing_idx": None,
        "nl_wizard_adding_rule": False,
        "nl_wizard_suggested_rules": [],
        "nl_wizard_loaded_rules": [],
    }
    for _k, _v in _NL_DEFAULTS.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    _nl_step = st.session_state.nl_step

    # On landing the Confirm/deploy step (3), scroll to the top so the summary + cost preview are
    # seen first — otherwise the page keeps the previous step's scroll and lands on the
    # "Confirm & Deploy" button at the bottom. One-shot: only on the transition INTO step 3.
    if _nl_step == 3 and st.session_state.get("_nl_prev_step") != 3:
        _scroll_to_top()
    st.session_state["_nl_prev_step"] = _nl_step

    # ── Step progress bar ─────────────────────────────────────────────────
    _STEP_LABELS = ["📝 Describe", "⚙️ Fields", "📋 Rules", "✅ Confirm", "🚀 Deploy"]
    _prog_cols = st.columns(len(_STEP_LABELS))
    for _pi, (_pc, _pl) in enumerate(zip(_prog_cols, _STEP_LABELS)):
        _is_active = _nl_step == _pi
        _is_done   = _nl_step > _pi
        _color = "#38bdf8" if _is_active else ("#4ade80" if _is_done else "#334155")
        _bg    = "rgba(56,189,248,0.10)" if _is_active else ("rgba(74,222,128,0.07)" if _is_done else "transparent")
        _fw    = "700" if _is_active else "400"
        with _pc:
            st.markdown(
                f'<div style="text-align:center;padding:0.35rem 0.2rem;background:{_bg};'
                f'border-radius:8px;border:1px solid {_color}33;">'
                f'<span style="color:{_color};font-size:0.76rem;font-weight:{_fw};">{_pl}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown("<div style='margin-top:1.1rem;'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 0 — Natural Language Input
    # ════════════════════════════════════════════════════════════════════════
    if _nl_step == 0:
        st.markdown(
            '<p style="color:#94a3b8;margin-bottom:0.5rem;">'
            'Describe your data pipeline in plain English.</p>',
            unsafe_allow_html=True,
        )
        _desc = st.text_area(
            "Pipeline description",
            value=st.session_state.nl_description,
            placeholder=(
                'e.g. "Daily sales data from PostgreSQL in Frankfurt to AWS S3, '
                'with PII masking on customer emails"'
            ),
            height=120,
            label_visibility="collapsed",
            key="nl_wizard_desc_textarea",
        )
        _s0c1, _ = st.columns([1, 3])
        with _s0c1:
            if st.button("Continue →", key="nl_wizard_step0_continue", type="primary",
                         disabled=not _desc.strip()):
                with st.spinner("Extracting intent from your description…"):
                    from utils.nlp_parser import _extract_intent
                    _intent = _extract_intent(_desc.strip())
                # Relevance gate — reject off-topic input instead of advancing into the wizard
                # with a fabricated config. (Mirrors nlp_parser.check_pipeline_request.)
                if not _intent.get("is_pipeline_request", True):
                    st.error(
                        "🚫 " + (_intent.get("rejection_reason")
                                 or "This doesn't look like a data-pipeline request. Describe a "
                                    "source table and a cloud destination (e.g. \"Postgres orders "
                                    "table to S3 daily as Parquet\").")
                    )
                else:
                    st.session_state.nl_description     = _desc.strip()
                    st.session_state.nl_intent          = _intent
                    st.session_state.nl_rules_initialized = False
                    # NEW authoring session — drop state carried over from a previous pipeline so
                    # the wizard starts fresh: stale rules/edits would otherwise make the Rules
                    # step auto-open the EDITOR (with the old rule) instead of the source picker,
                    # and stale field answers would pre-fill the wrong values. The Rules step then
                    # re-seeds from THIS description's extracted rules, else shows the picker.
                    st.session_state.nl_rules        = []
                    st.session_state.nl_rules_view   = None
                    st.session_state.nl_answers      = {}
                    st.session_state.nl_step            = 1
                    st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # STEP 1 — Field Clarification
    # ════════════════════════════════════════════════════════════════════════
    elif _nl_step == 1:
        from utils.nlp_parser import FIELD_OPTIONS as _FO
        st.markdown(
            '<p style="color:#94a3b8;margin-bottom:1rem;">'
            'Review and confirm every pipeline field below.</p>',
            unsafe_allow_html=True,
        )

        _intent_raw = st.session_state.nl_intent
        _prev_ans   = st.session_state.nl_answers  # non-empty when editing from step 3

        def _pre1(field, default=""):
            return _prev_ans.get(field) or _intent_raw.get(field) or default

        def _lbl(text):
            st.markdown(
                f'<p style="color:#94a3b8;font-size:0.84rem;margin:0.8rem 0 0.2rem;">{text}</p>',
                unsafe_allow_html=True,
            )

        _lbl("Pipeline name (slug)")
        _slug_v = st.text_input("slug", value=_pre1("pipeline_slug"),
                                placeholder="e.g. eu_sales",
                                label_visibility="collapsed", key="nl_wizard_f_slug")

        _domain_opts = _FO["data_domain"]
        _domain_pre  = _pre1("data_domain", "sales")
        _domain_idx  = _domain_opts.index(_domain_pre) if _domain_pre in _domain_opts else 0
        _lbl("Data domain")
        _domain_v = st.radio("domain", _domain_opts, index=_domain_idx, horizontal=True,
                             label_visibility="collapsed", key="nl_wizard_f_domain")

        _db_opts = _FO["source_db_type"]
        _db_pre  = _pre1("source_db_type", "postgres")
        _db_idx  = _db_opts.index(_db_pre) if _db_pre in _db_opts else 0
        _lbl("Source database type")
        _db_v = st.radio("dbtype", _db_opts, index=_db_idx, horizontal=True,
                         label_visibility="collapsed", key="nl_wizard_f_dbtype")

        _lbl("Source table name")
        _table_v = st.text_input("table", value=_pre1("source_table"),
                                 placeholder="e.g. raw_orders",
                                 label_visibility="collapsed", key="nl_wizard_f_table")

        _lbl("Target cloud")
        st.markdown('<p style="color:#f97316;font-weight:600;margin:0;">🟠 AWS (S3 + EKS)</p>',
                    unsafe_allow_html=True)
        _cloud_v = "aws"

        _freq_opts = _FO["frequency"]
        _freq_pre  = _pre1("frequency", "daily")
        _freq_idx  = _freq_opts.index(_freq_pre) if _freq_pre in _freq_opts else 0
        _lbl("Frequency")
        _freq_v = st.radio("freq", _freq_opts, index=_freq_idx, horizontal=True,
                           label_visibility="collapsed", key="nl_wizard_f_freq")

        _lbl("Owner team")
        _owner_v = st.text_input("owner", value=_pre1("owner_team", "analytics_team"),
                                 placeholder="e.g. analytics_team",
                                 label_visibility="collapsed", key="nl_wizard_f_owner")

        st.markdown("<hr style='margin:1.2rem 0;'>", unsafe_allow_html=True)
        _s1c1, _s1c2, _ = st.columns([1, 1, 3])
        with _s1c1:
            if st.button("Confirm fields →", key="nl_wizard_step1_confirm", type="primary"):
                _errs = []
                if not _slug_v.strip():
                    _errs.append("Pipeline name is required.")
                if not _table_v.strip():
                    _errs.append("Source table is required.")
                if _errs:
                    for _e in _errs:
                        st.error(_e)
                else:
                    st.session_state.nl_answers = {
                        "pipeline_slug": _slug_v.strip(),
                        "data_domain":   _domain_v,
                        "source_db_type": _db_v,
                        "source_table":  _table_v.strip(),
                        "target_cloud":  _cloud_v,
                        "frequency":     _freq_v,
                        "owner_team":    _owner_v.strip() or "analytics_team",
                    }
                    _dest = st.session_state.pop("nl_back_to_step", None) or 2
                    st.session_state.nl_rules_view = None  # auto-decide editor/picker on entry
                    st.session_state.nl_step = _dest
                    st.rerun()
        with _s1c2:
            if st.button("← Back", key="nl_wizard_step1_back", type="secondary"):
                st.session_state.nl_step = 0
                st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2 — Business Rules
    # ════════════════════════════════════════════════════════════════════════
    elif _nl_step == 2:
        st.markdown(
            '<p style="color:#94a3b8;margin-bottom:1rem;">Review and edit business rules.</p>',
            unsafe_allow_html=True,
        )

        _raw_rules = st.session_state.nl_intent.get("business_rules", [])
        _domain    = st.session_state.nl_answers.get("data_domain", "other")

        # STALE-RULES GUARD (path-independent): the held rules are stamped with the
        # description they were seeded from. If that differs from the CURRENT description,
        # they belong to a previous/other pipeline (reached here via "Edit fields", Back, or
        # an un-deployed prior session whose nl_ keys were never cleared) — drop them so THIS
        # run re-seeds from its own extraction (else shows the source picker). Same
        # description → rules/edits are preserved. This generalises the step-0 reset to
        # EVERY entry path, not just "re-describe".
        if st.session_state.get("nl_rules_desc") != st.session_state.nl_description:
            st.session_state.nl_rules             = []
            st.session_state.nl_rules_initialized = False
            st.session_state.nl_rules_view        = None
            st.session_state.nl_rules_desc        = st.session_state.nl_description

        if not st.session_state.nl_rules_initialized:
            # Do NOT auto-apply the NL-extracted rules. The user asked to ALWAYS land on the source
            # picker (build own / suggest / file / none) and choose explicitly — never to arrive at
            # pre-filled rules they didn't ask for (e.g. the auto PII flags derived from the schema).
            # The extracted rules are not lost: they are offered on demand as the "Suggest" proposal
            # (_raw_rules) below. Rules ALREADY held (a picker choice followed by a step-3 re-entry)
            # are preserved untouched — only the first-entry auto-seed is removed.
            st.session_state.nl_rules_initialized   = True
            st.session_state.nl_wizard_editing_idx  = None
            st.session_state.nl_wizard_adding_rule  = False
            st.session_state.nl_wizard_suggested_rules = []
            st.session_state.nl_wizard_loaded_rules = []

        # Auto-decide the sub-step view on entry: straight to the editor if we already have
        # rules (e.g. extracted from the NL description), otherwise the source picker.
        if st.session_state.nl_rules_view is None:
            st.session_state.nl_rules_view = "editor" if st.session_state.nl_rules else "picker"

        _ACTION_OPTS = ["DROP_RECORD", "EXCLUDE_AND_LOG", "FLAG_AS_SUSPICIOUS", "DEFAULT_VALUE"]
        _ACTION_COLORS = {
            "DROP_RECORD": "#f87171", "EXCLUDE_AND_LOG": "#fbbf24",
            "FLAG_AS_SUSPICIOUS": "#f97316", "DEFAULT_VALUE": "#4ade80",
        }

        def _rule_edit_form(prefix, existing=None):
            existing = existing or {}
            _ri = st.text_input("Rule ID", value=existing.get("rule_id", ""),
                                key=f"{prefix}_id")
            st.markdown(
                '<p style="color:#475569;font-size:0.75rem;margin:0.2rem 0 0.1rem;">'
                "Description — plain words (the agent reads this alongside the condition)</p>",
                unsafe_allow_html=True,
            )
            _rr = st.text_input("Description", value=existing.get("reason", ""),
                                key=f"{prefix}_reason")
            _rc = st.text_input("Target column", value=existing.get("target_column", ""),
                                key=f"{prefix}_col")
            st.markdown(
                '<p style="color:#475569;font-size:0.75rem;margin:0.2rem 0 0.1rem;">'
                "Condition — pandas expression, e.g. df['col'].notna(). "
                "Or leave it empty and just write the Description above — we'll derive it.</p>",
                unsafe_allow_html=True,
            )
            _rk = st.text_input("Condition", value=existing.get("condition", ""),
                                 key=f"{prefix}_cond")
            _ai = _ACTION_OPTS.index(existing.get("action", "DROP_RECORD")) \
                  if existing.get("action") in _ACTION_OPTS else 0
            _ra = st.radio("Action", _ACTION_OPTS, index=_ai, horizontal=True,
                           key=f"{prefix}_action")
            return {"rule_id": _ri, "target_column": _rc,
                    "condition": _rk, "action": _ra, "reason": _rr}

        def _finalize_rule(rule: dict) -> dict:
            # A manually-entered rule with plain words but NO pandas condition is filled the
            # SAME way a free-text rules file is — derive the condition from the description via
            # the LLM. If the user already typed a condition, keep it exactly as-is.
            if rule.get("condition", "").strip() or not rule.get("reason", "").strip():
                return rule
            try:
                from utils.nlp_parser import synthesize_rule
                return synthesize_rule(rule["reason"], rule.get("target_column", ""),
                                       rule.get("action", ""))
            except Exception:
                return rule  # fall back to as-entered; the architect still reads the description

        # ─────────────────────────────────────────────────────────────────────
        # EDITABLE rules (view == "editor"). Rules reach here from ANY source — extracted from
        # the NL description, loaded from a file, or domain suggestions. "← Back" steps to the
        # source picker (view == "picker"), keeping the rules; a second "← Back" there → Fields.
        # ─────────────────────────────────────────────────────────────────────
        if st.session_state.nl_rules_view == "editor":
            _rules      = st.session_state.nl_rules
            _edit_idx   = st.session_state.nl_wizard_editing_idx

            st.markdown(
                f'<p style="color:#4ade80;font-size:0.82rem;margin-bottom:0.5rem;">'
                f'✓ {len(_rules)} rule(s) — review, edit, add or remove below.</p>',
                unsafe_allow_html=True,
            )

            for _ri_i, _ri_r in enumerate(_rules):
                if _edit_idx == _ri_i:
                    # ── Inline edit form ─────────────────────────────────────
                    st.markdown(
                        f'<div style="background:rgba(56,189,248,0.08);border:1px solid '
                        f'rgba(56,189,248,0.3);border-radius:10px;padding:0.8rem 1rem;margin:0.4rem 0;">'
                        f'<p style="color:#38bdf8;font-size:0.82rem;font-weight:600;margin:0 0 0.6rem;">'
                        f'✏️ Editing rule {_ri_i + 1}</p></div>',
                        unsafe_allow_html=True,
                    )
                    _edited = _rule_edit_form(f"nl_wizard_edit_{_ri_i}", existing=_ri_r)
                    _ef1, _ef2, _ = st.columns([1, 1, 3])
                    with _ef1:
                        if st.button("💾 Save", key=f"nl_wizard_rule_save_{_ri_i}",
                                     type="primary"):
                            with st.spinner("Finalising rule…"):
                                st.session_state.nl_rules[_ri_i] = _finalize_rule(_edited)
                            st.session_state.nl_wizard_editing_idx = None
                            st.rerun()
                    with _ef2:
                        if st.button("✗ Cancel", key=f"nl_wizard_rule_cancel_{_ri_i}",
                                     type="secondary"):
                            st.session_state.nl_wizard_editing_idx = None
                            st.rerun()
                else:
                    # ── Rule card ─────────────────────────────────────────────
                    with st.expander(
                        f"Rule {_ri_i + 1}: {_ri_r.get('rule_id', '?')}", expanded=True
                    ):
                        _card1, _card2, _card3 = st.columns([4, 2, 1])
                        with _card1:
                            _ac = _ACTION_COLORS.get(_ri_r.get("action", ""), "#94a3b8")
                            # The plain-words rationale (your file's `description`, mapped to
                            # `reason`) — shown so the rule isn't just column/action/condition.
                            _reason_html = (
                                f'<p style="margin:0.25rem 0 0;color:#cbd5e1;font-size:0.78rem;'
                                f'font-style:italic;">📝 {_ri_r.get("reason","")}</p>'
                                if _ri_r.get("reason") else ''
                            )
                            st.markdown(
                                f'<p style="margin:0;color:#94a3b8;font-size:0.8rem;">'
                                f'<b style="color:#e2e8f0;">column:</b> {_ri_r.get("target_column","")}'
                                f' &nbsp;|&nbsp; '
                                f'<span style="color:{_ac};font-weight:600;">{_ri_r.get("action","")}</span>'
                                f'</p>'
                                f'{_reason_html}'
                                f'<p style="margin:0.2rem 0 0;color:#475569;font-size:0.75rem;">'
                                f'<code style="color:#7dd3fc;">{_ri_r.get("condition","")}</code></p>',
                                unsafe_allow_html=True,
                            )
                        with _card3:
                            _btn1, _btn2 = st.columns(2)
                            with _btn1:
                                if st.button("✏️", key=f"nl_wizard_rule_edit_{_ri_i}",
                                             help="Edit rule"):
                                    st.session_state.nl_wizard_editing_idx = _ri_i
                                    st.rerun()
                            with _btn2:
                                if st.button("🗑️", key=f"nl_wizard_rule_del_{_ri_i}",
                                             help="Remove rule"):
                                    st.session_state.nl_rules.pop(_ri_i)
                                    st.session_state.nl_wizard_editing_idx = None
                                    st.rerun()

            # ── Add a rule — two side-by-side methods as TABS (no scrolling to find them) ──
            st.markdown(
                '<p style="color:#4ade80;font-size:0.82rem;font-weight:600;margin:0.8rem 0 0.3rem;">'
                '+ Add a rule</p>',
                unsafe_allow_html=True,
            )
            _tab_fields, _tab_nl = st.tabs(["🧱 Fields", "🗣️ Plain English"])
            with _tab_fields:
                _new_r = _rule_edit_form("nl_wizard_add")
                if st.button("💾 Add rule", key="nl_wizard_rule_add_save", type="primary"):
                    with st.spinner("Finalising rule…"):
                        st.session_state.nl_rules.append(_finalize_rule(_new_r))
                    st.rerun()
            with _tab_nl:
                st.markdown(
                    '<p style="color:#94a3b8;font-size:0.78rem;margin:0 0 0.3rem;">'
                    'Write all your rules in plain English — parsed into structured rules, '
                    'the same way an uploaded file is.</p>',
                    unsafe_allow_html=True,
                )
                _nl_rules_txt = st.text_area(
                    "Plain-English rules",
                    placeholder=("e.g. Drop orders where unit_price is 0 or negative. Exclude "
                                 "future-dated orders. Replace unknown currency with EUR. "
                                 "Flag quantity over 1000 as suspicious."),
                    key="nl_wizard_build_nl", height=120, label_visibility="collapsed",
                )
                if (st.button("✨ Parse to rules", key="nl_wizard_build_nl_btn", type="primary")
                        and _nl_rules_txt.strip()):
                    with st.spinner("Interpreting your rules…"):
                        from utils.nlp_parser import parse_rules_from_content
                        try:
                            _parsed = parse_rules_from_content(_nl_rules_txt)
                            st.session_state.nl_rules.extend(_parsed)
                            st.success(f"Added {len(_parsed)} rule(s) — review them above.")
                            st.rerun()
                        except Exception as _e:
                            st.error(f"Could not interpret your rules: {_e}")

            st.markdown("<hr style='margin:1.2rem 0;'>", unsafe_allow_html=True)
            _s2a1, _s2a2, _ = st.columns([1, 1, 3])
            with _s2a1:
                if st.button("Continue →", key="nl_wizard_step2_continue", type="primary"):
                    _dest = st.session_state.pop("nl_back_to_step", None) or 3
                    st.session_state.nl_step = _dest
                    st.rerun()
            with _s2a2:
                # ONE hierarchical "← Back": from the editor it steps to the source picker
                # (keeping the rules) so you can change how you got them; from the picker it
                # goes to Fields (handled below).
                if st.button("← Back", key="nl_wizard_step2_back", type="secondary"):
                    st.session_state.nl_rules_view = "picker"
                    st.session_state.nl_wizard_editing_idx = None
                    st.session_state.nl_wizard_adding_rule = False
                    st.rerun()

        # ─────────────────────────────────────────────────────────────────────
        # No rules YET — pick a source (domain suggestions / file / none). Whatever the
        # source, the rules land in the editable list above for review/edit/add/remove.
        # ─────────────────────────────────────────────────────────────────────
        else:
            if st.session_state.nl_rules:
                # Stepped back here from the editor (rules already exist). Offer a forward path
                # that KEEPS them, next to re-picking a source (which replaces them).
                st.markdown(
                    f'<p style="color:#94a3b8;font-size:0.85rem;margin-bottom:0.5rem;">'
                    f'You have {len(st.session_state.nl_rules)} rule(s). Keep & edit them, or '
                    f'pick a different source below (this replaces them).</p>',
                    unsafe_allow_html=True,
                )
                if st.button("✏️ Keep & edit my rules", key="nl_wizard_keep_rules",
                             type="primary"):
                    st.session_state.nl_rules_view = "editor"
                    st.rerun()
            else:
                st.markdown(
                    '<p style="color:#fbbf24;font-size:0.85rem;margin-bottom:0.8rem;">'
                    'No business rules yet — add them from a source below '
                    '(you can edit, add or remove them afterwards).</p>',
                    unsafe_allow_html=True,
                )

            _rule_mode = st.radio(
                "How to handle rules?",
                ["💡 Suggest rules for my domain",
                 "📁 Load from file (.md / .yaml / .json)",
                 "✍️ Build my own rules",
                 "⏭️ Continue without rules"],
                key="nl_wizard_rule_mode_radio",
                label_visibility="collapsed",
            )

            if _rule_mode == "⏭️ Continue without rules":
                # Advance immediately on selection
                st.session_state.nl_rules = []
                _dest = st.session_state.pop("nl_back_to_step", None) or 3
                st.session_state.nl_step = _dest
                st.rerun()

            elif _rule_mode == "✍️ Build my own rules":
                # Author rules from scratch: open the (empty) editor, whose "+ Add a rule" area
                # offers BOTH methods as tabs — 🧱 Fields and 🗣️ Plain English.
                st.session_state.nl_rules_view = "editor"
                st.rerun()

            elif _rule_mode == "💡 Suggest rules for my domain":
                if st.button("✨ Generate suggestions", key="nl_wizard_suggest_btn",
                             type="primary"):
                    with st.spinner(f"Generating rules for '{_domain}' domain…"):
                        # Prefer the rules extracted from the user's OWN description (schema-aware —
                        # e.g. the auto PII flags); fall back to generic domain rules only if the
                        # description yielded none. Either way the user PICKS which to keep below —
                        # nothing is auto-applied.
                        from utils.nlp_parser import suggest_rules_for_domain
                        st.session_state.nl_wizard_suggested_rules = \
                            list(_raw_rules) or suggest_rules_for_domain(_domain)
                    st.rerun()

                _suggestions = st.session_state.nl_wizard_suggested_rules
                if _suggestions:
                    st.markdown(
                        f'<p style="color:#4ade80;font-size:0.82rem;margin:0.5rem 0 0.3rem;">'
                        f'✓ {len(_suggestions)} suggestion(s) — pick which to keep:</p>',
                        unsafe_allow_html=True,
                    )
                    _keep_mask = []
                    for _si, _sr in enumerate(_suggestions):
                        _ac = _ACTION_COLORS.get(_sr.get("action", ""), "#94a3b8")
                        _k = st.checkbox(
                            f"{_si + 1}. **{_sr.get('rule_id','')}** — "
                            f"`{_sr.get('target_column','')}` — "
                            f"{_sr.get('action','')}",
                            value=True,
                            key=f"nl_wizard_suggest_keep_{_si}",
                        )
                        _keep_mask.append(_k)
                    if st.button("Use selected →", key="nl_wizard_suggest_use",
                                 type="primary"):
                        st.session_state.nl_rules = [
                            _sr for _sr, _k in zip(_suggestions, _keep_mask) if _k
                        ]
                        st.session_state.nl_wizard_suggested_rules = []
                        st.session_state.nl_rules_view = "editor"
                        st.rerun()  # → the editor renders them (editable)

            elif _rule_mode == "📁 Load from file (.md / .yaml / .json)":
                _up_rules = st.file_uploader(
                    "Upload rules file",
                    type=["md", "yaml", "yml", "json", "txt"],
                    key="nl_wizard_rules_upload",
                    label_visibility="collapsed",
                )
                if _up_rules:
                    if st.button("Parse file →", key="nl_wizard_rules_parse_btn",
                                 type="primary"):
                        with st.spinner("Parsing rules from file…"):
                            from utils.nlp_parser import parse_rules_from_content
                            _content = _up_rules.read().decode("utf-8", errors="replace")
                            st.session_state.nl_wizard_loaded_rules = \
                                parse_rules_from_content(_content)
                        st.rerun()

                _loaded = st.session_state.nl_wizard_loaded_rules
                if _loaded:
                    st.markdown(
                        f'<p style="color:#4ade80;font-size:0.82rem;margin:0.5rem 0 0.3rem;">'
                        f'✓ {len(_loaded)} rule(s) parsed from file.</p>',
                        unsafe_allow_html=True,
                    )
                    for _li, _lr in enumerate(_loaded):
                        st.markdown(
                            f'<div style="border-left:2px solid rgba(56,189,248,0.3);'
                            f'padding:0.25rem 0.7rem;margin:0.2rem 0;font-size:0.8rem;">'
                            f'<b style="color:#e2e8f0;">{_lr.get("rule_id","")}</b>'
                            f' — {_lr.get("target_column","")} — '
                            f'<span style="color:{_ACTION_COLORS.get(_lr.get("action",""),"#94a3b8")};">'
                            f'{_lr.get("action","")}</span></div>',
                            unsafe_allow_html=True,
                        )
                    if st.button("Use these rules →", key="nl_wizard_loaded_use",
                                 type="primary"):
                        st.session_state.nl_rules = list(_loaded)
                        st.session_state.nl_wizard_loaded_rules = []
                        st.session_state.nl_rules_view = "editor"
                        st.rerun()  # → the editor renders them (editable)

            st.markdown("<hr style='margin:1.2rem 0;'>", unsafe_allow_html=True)
            if st.button("← Back", key="nl_wizard_step2b_back", type="secondary"):
                st.session_state.nl_step = 1
                st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # STEP 3 — Summary & Confirmation
    # ════════════════════════════════════════════════════════════════════════
    elif _nl_step == 3:
        _ans   = st.session_state.nl_answers
        _rules = st.session_state.nl_rules
        _cf    = {"aws": "🟠 AWS"}
        _cloud_label = _cf.get(_ans.get("target_cloud", ""), _ans.get("target_cloud", ""))

        # ── Rules section HTML ────────────────────────────────────────────
        if _rules:
            _rules_header = f'Business Rules ({len(_rules)})'
            _rules_body   = "".join(
                f'<div style="padding:0.15rem 0;color:#94a3b8;font-size:0.79rem;">'
                f'• <b style="color:#e2e8f0;">{_r.get("rule_id","")}</b>'
                f' &nbsp;—&nbsp; {_r.get("target_column","")}'
                f' &nbsp;—&nbsp; <span style="color:#f97316;">{_r.get("action","")}</span>'
                f'</div>'
                for _r in _rules
            )
        else:
            _rules_header = "Business Rules"
            _rules_body   = '<span style="color:#334155;">none</span>'

        st.markdown(
            f'<div style="background:rgba(10,18,40,0.85);border:2px solid rgba(56,189,248,0.35);'
            f'border-radius:14px;padding:1.4rem 1.6rem;margin:0.4rem 0 1rem;">'
            f'<p style="color:#475569;font-size:0.7rem;text-transform:uppercase;'
            f'letter-spacing:0.09em;margin:0 0 0.9rem;">Pipeline Configuration Summary</p>'
            f'<table style="width:100%;border-collapse:collapse;font-size:0.84rem;">'
            f'<tr><td style="color:#64748b;padding:0.22rem 0;width:150px;">Pipeline</td>'
            f'<td style="color:#e2e8f0;font-weight:600;">{_ans.get("pipeline_slug","")}</td></tr>'
            f'<tr><td style="color:#64748b;padding:0.22rem 0;">Domain</td>'
            f'<td style="color:#e2e8f0;">{_ans.get("data_domain","")}</td></tr>'
            f'<tr><td style="color:#64748b;padding:0.22rem 0;">Database</td>'
            f'<td style="color:#e2e8f0;">{_ans.get("source_db_type","")} &nbsp;→&nbsp; '
            f'<span style="color:#7dd3fc;">table: {_ans.get("source_table","")}</span></td></tr>'
            f'<tr><td style="color:#64748b;padding:0.22rem 0;">Cloud</td>'
            f'<td style="color:#e2e8f0;">{_cloud_label}</td></tr>'
            f'<tr><td style="color:#64748b;padding:0.22rem 0;">Frequency</td>'
            f'<td style="color:#e2e8f0;">{_ans.get("frequency","")}</td></tr>'
            f'<tr><td style="color:#64748b;padding:0.22rem 0;">Owner</td>'
            f'<td style="color:#e2e8f0;">{_ans.get("owner_team","")}</td></tr>'
            f'<tr><td style="color:#64748b;padding:0.3rem 0 0.1rem;vertical-align:top;">'
            f'{_rules_header}</td>'
            f'<td style="padding:0.3rem 0 0.1rem;">{_rules_body}</td></tr>'
            f'</table></div>',
            unsafe_allow_html=True,
        )

        # ── Execution plan — what will be created (illustrative, no run) ──
        st.markdown(
            '<p style="color:#475569;font-size:0.7rem;text-transform:uppercase;'
            'letter-spacing:0.09em;margin:0.4rem 0 0.3rem;">'
            'Execution plan — what will be created</p>',
            unsafe_allow_html=True,
        )
        _render_execution_plan(
            "aws",
            f"pipe_{_ans.get('pipeline_slug', 'pipeline')}_to_s3",
        )

        # ── Cost preview (shown BEFORE the explicit deploy) ───────────────
        st.markdown(
            '<p style="color:#475569;font-size:0.7rem;text-transform:uppercase;'
            'letter-spacing:0.09em;margin:0.4rem 0 0.3rem;">'
            f'Estimated monthly cost &nbsp;·&nbsp; your pick: {_cloud_label}</p>',
            unsafe_allow_html=True,
        )
        _render_cost_charts(key="nl")

        # ── Edit buttons ──────────────────────────────────────────────────
        st.markdown(
            '<p style="color:#475569;font-size:0.8rem;margin:0 0 0.4rem;">'
            'Need to change something?</p>',
            unsafe_allow_html=True,
        )
        _e1, _e2, _ = st.columns([1, 1, 3])
        with _e1:
            if st.button("✏️ Edit fields", key="nl_wizard_step3_edit_fields",
                         type="secondary"):
                st.session_state.nl_back_to_step = 3
                st.session_state.nl_step = 1
                st.rerun()
        with _e2:
            if st.button("📋 Edit rules", key="nl_wizard_step3_edit_rules",
                         type="secondary"):
                st.session_state.nl_back_to_step    = 3
                st.session_state.nl_rules_initialized = False
                st.session_state.nl_rules_view = None  # auto-decide editor/picker on entry
                st.session_state.pop("nl_wizard_rule_mode_radio", None)
                st.session_state.nl_step = 2
                st.rerun()

        st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)
        _a1, _a2, _ = st.columns([1, 1, 3])
        with _a1:
            if st.button("✅ Confirm & Deploy", key="nl_wizard_step3_confirm",
                         type="primary"):
                st.session_state.nl_summary_ok = True
                st.session_state.nl_step = 4
                st.rerun()
        with _a2:
            if st.button("✗ Cancel", key="nl_wizard_step3_cancel", type="secondary"):
                for _ck in [k for k in st.session_state if k.startswith("nl_")]:
                    del st.session_state[_ck]
                st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # STEP 4 — Deploy
    # ════════════════════════════════════════════════════════════════════════
    elif _nl_step == 4:
        st.markdown(
            '<p style="color:#4ade80;font-size:0.9rem;margin-bottom:0.8rem;">'
            '🚀 Building and launching your pipeline…</p>',
            unsafe_allow_html=True,
        )
        _deploy_cloud = st.session_state.nl_answers.get("target_cloud") or "aws"
        _ready, _why = _bootstrap_ready(_deploy_cloud)
        if not _ready:
            st.error("🚫 " + _why)
            if st.button("← Back to summary", key="nl_wizard_step4_notready_back",
                         type="secondary"):
                st.session_state.nl_step = 3
                st.rerun()
            st.stop()
        with st.spinner("Building pipeline configuration…"):
            try:
                from utils.nlp_parser import _build_from_answers
                _pc, _dc, _rc, _ic, _pid, _tk = _build_from_answers(
                    st.session_state.nl_answers,
                    st.session_state.nl_rules,
                    description=st.session_state.nl_description,
                )
            except Exception as _ex:
                st.error(f"Could not build pipeline config: {_ex}")
                if st.button("← Back to summary", key="nl_wizard_step4_back",
                             type="secondary"):
                    st.session_state.nl_step = 3
                    st.rerun()
                st.stop()

        _start_run(_pc, _dc, _rc, _ic, _pid, _tk)
        # Reset wizard so user can launch another pipeline later
        for _rk in [k for k in st.session_state if k.startswith("nl_")]:
            del st.session_state[_rk]
        st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # Advanced: Architecture Analysis (preserved below wizard)
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    with st.expander("🔍 Advanced: Architecture Analysis", expanded=False):
        st.markdown(
            '<p style="color:#94a3b8;margin-bottom:0.5rem;">'
            'See how your workload maps onto the AWS data-lake stack, with a cost estimate.</p>',
            unsafe_allow_html=True,
        )
        _adv_desc = st.text_area(
            "Description for analysis",
            value=st.session_state.get("nl_description", ""),
            placeholder="Describe your pipeline for architecture analysis…",
            height=80,
            label_visibility="collapsed",
            key="nl_adv_desc",
        )
        _adv_c1, _ = st.columns([1, 3])
        with _adv_c1:
            _analyze_btn = st.button("🔍 Analyze & Plan", type="primary",
                                     key="nl_wizard_analyze_btn",
                                     disabled=not _adv_desc.strip())
        if _analyze_btn and _adv_desc.strip():
            from utils.nlp_parser import check_pipeline_request
            _adv_ok, _adv_why = check_pipeline_request(_adv_desc.strip())
            if not _adv_ok:
                st.error("🚫 " + (_adv_why or "This doesn't look like a data-pipeline request."))
            else:
                with st.spinner("Analyzing requirements and costing the AWS stack…"):
                    from utils.architecture_advisor import analyze
                    st.session_state.arch_report = analyze(_adv_desc)

        _adv_report = st.session_state.get("arch_report")
        if _adv_report:
            _render_arch_report(_adv_report)

        _adv_cloud = "aws"
        if st.button(
            "🚀 Deploy on AWS",
            key="nl_wizard_launch_arch",
            type="primary",
            disabled=not _adv_report,
        ):
            _arch_ready, _arch_why = _bootstrap_ready(_adv_cloud)
            if not _arch_ready:
                st.error("🚫 " + _arch_why)
                st.stop()
            if st.session_state.get("nl_answers"):
                with st.spinner(f"Building config for {_adv_cloud.upper()}…"):
                    from utils.nlp_parser import _build_from_answers
                    _apc, _adc, _arc, _aic, _apid, _atk = _build_from_answers(
                        st.session_state.nl_answers,
                        st.session_state.get("nl_rules", []),
                        description=st.session_state.get("nl_description", ""),
                        cloud_override=_adv_cloud,
                    )
                _start_run(_apc, _adc, _arc, _aic, _apid, _atk)
                st.rerun()
            else:
                st.warning("Complete the wizard steps first to set pipeline fields.")

# launch_nl is no longer a button — wizard handles deploy directly in step 4
launch_nl = False

with tab_existing:
    st.markdown(
        '<p style="color:#94a3b8; margin-bottom:0.5rem;">Run a pre-built reference pipeline. Structured example: '
        'EU sales orders from RDS PostgreSQL → cleaned Parquet on S3 (partitioned by run_date) '
        '→ Trino → Grafana.</p>',
        unsafe_allow_html=True,
    )
    _SAMPLE_PIPELINES = {
        "eu_sales": "Structured: RDS PostgreSQL table → Parquet on S3",
        "shop_orders": "Semi-structured: MongoDB order documents (nested customer + items array) → flattened Parquet on S3",
    }
    existing_choice = st.selectbox("Select a sample pipeline", list(_SAMPLE_PIPELINES),
                                   format_func=lambda k: f"{k} — {_SAMPLE_PIPELINES[k]}")
    st.markdown(
        '<span style="color:#64748b;font-size:0.85rem;">Target: 🟠 AWS (EKS + S3)</span>',
        unsafe_allow_html=True,
    )
    _render_cost_panel(selected_cloud="aws")
    with st.expander("⚙️ Business Rules", expanded=False):
        _render_rules_selector(key_prefix="ex")
    launch_existing = st.button("▶️ Run Pipeline", type="primary", key="launch_existing")

# ── Trino SQL tab ───────────────────────────────────────────────────────────
with tab_trino:
    from utils.trino_client import SAMPLE_QUERY, run_query

    st.markdown(
        '<p style="color:#94a3b8;margin-bottom:0.8rem;">'
        'Query the <b style="color:#f97316;">S3 Parquet data lake</b> with SQL through Trino. '
        'The table is partitioned by <code>run_date</code>, so date filters only read the '
        'matching S3 prefixes (partition pruning).</p>',
        unsafe_allow_html=True,
    )

    sql_input = st.text_area(
        "SQL",
        value=SAMPLE_QUERY,
        height=260,
        label_visibility="collapsed",
    )

    run_trino = st.button("▶️ Run Query", type="primary", key="run_trino")

    if run_trino:
        with st.spinner("Executing on Trino…"):
            df, is_live = run_query(sql_input)

        mode_html = (
            '<span style="color:#4ade80;font-size:0.8rem;">● Live Trino connection</span>'
            if is_live else
            '<span style="color:#fbbf24;font-size:0.8rem;">⚡ Demo mode — set TRINO_HOST to connect a live cluster</span>'
        )
        st.markdown(mode_html, unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:#64748b;font-size:0.82rem;margin:0.3rem 0 0.8rem;">'
            f'{len(df):,} rows returned</p>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )
        if not is_live:
            st.markdown(
                '<div style="background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.2);'
                'border-radius:10px;padding:0.7rem 1rem;margin-top:0.8rem;">'
                '<p style="margin:0;color:#94a3b8;font-size:0.82rem;">'
                '💡 To connect a real Trino cluster, add <code style="color:#7dd3fc;">TRINO_HOST=your-host</code> '
                'to your <code style="color:#7dd3fc;">.env</code>. '
                '(port-forward with <code>kubectl port-forward svc/trino 8080:8080 -n analytics</code>). '
                'The <code>hive</code> catalog (Glue metastore over S3) is deployed by the agents.</p>'
                '</div>',
                unsafe_allow_html=True,
            )

# ── Observability tab ───────────────────────────────────────────────────────
with tab_obs:
    from utils.observability import get_pipeline_summary, get_hourly_throughput, get_partition_breakdown

    last_run_id = st.session_state.get("pipeline_meta", {}).get("pipeline_id", "eu-sales-pipeline")

    st.markdown(
        f'<p style="color:#94a3b8;font-size:0.85rem;margin-bottom:0.8rem;">'
        f'Monitoring: <code style="color:#38bdf8;">{last_run_id}</code>'
        f'{"" if st.session_state.get("pipeline_meta") else " &nbsp;·&nbsp; <em>launch a pipeline to see live metrics</em>"}'
        f'</p>',
        unsafe_allow_html=True,
    )

    summary = get_pipeline_summary(last_run_id)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Records Today",    f"{summary['records_today']:,}")
    m2.metric("Error Rate",       f"{summary['error_rate_pct']}%")
    m3.metric("Avg Latency",      f"{summary['avg_latency_ms']} ms")
    m4.metric("SLA",              f"{summary['sla_met_pct']}%")

    st.markdown(
        f'<p style="color:#334155;font-size:0.72rem;margin:0.2rem 0 1rem;">'
        f'Last run {summary["last_run_ago_min"]} min ago &nbsp;·&nbsp; '
        f'{summary["runs_today"]} runs today &nbsp;·&nbsp; '
        f'source: <em>{summary["source"]}</em></p>',
        unsafe_allow_html=True,
    )

    st.markdown("##### 📈 Hourly Throughput (last 24 h)")
    throughput_df = get_hourly_throughput(last_run_id)
    st.area_chart(
        throughput_df.set_index("time")[["records", "errors"]],
        color=["#38bdf8", "#f87171"],
        use_container_width=True,
        height=200,
    )

    st.markdown("##### 🗂️ Data-lake partitions (run_date) — rows written vs rejected by the rules")
    part_df = get_partition_breakdown(last_run_id)
    st.dataframe(
        part_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "run_date":      st.column_config.TextColumn("Partition (run_date)"),
            "rows_written":  st.column_config.NumberColumn("Rows written", format="%d"),
            "rows_rejected": st.column_config.NumberColumn("Rows rejected", format="%d"),
            "reject_pct":    st.column_config.NumberColumn("Reject %", format="%.2f%%"),
            "parquet_mb":    st.column_config.NumberColumn("Parquet size (MB)", format="%.1f"),
            "files":         st.column_config.NumberColumn("Files", format="%d"),
        },
    )

    if summary["source"] == "simulated":
        st.markdown(
            '<p style="color:#334155;font-size:0.72rem;margin-top:0.5rem;">'
            '* Simulated data — add <code>PROMETHEUS_URL</code> to .env to stream live metrics</p>',
            unsafe_allow_html=True,
        )

# launch_nl is always False — the wizard deploys directly inside tab_nl step 4.
# (_start_run lives next to _run_agent above — it MUST be defined before the
#  tab blocks: Streamlit executes the script top-to-bottom, so a call from the
#  wizard's step 4 would hit a NameError if the def came later in the file.)

if launch_existing:
    with st.spinner(f"Loading {existing_choice} config..."):
        try:
            from agents.constants import CONFIGS_DIR
            from utils.config_utils import load_pipeline_bundle
            from utils.prompt_utils import format_prompt
            from utils.file_utils import read_file

            pipeline_dir = os.path.join(CONFIGS_DIR, "pipelines")
            pipe_conf, db_conf, rules_conf, infra_conf = load_pipeline_bundle(
                os.getcwd(),
                os.path.join(pipeline_dir, f"{existing_choice}_pipeline.yaml"),
            )
            pipeline_id = pipe_conf.get("pipeline_id", existing_choice)
            ts  = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            task = format_prompt(
                read_file(os.path.join(pipeline_dir, f"{existing_choice}_objective.md")),
                project_id=f"{existing_choice.upper()}-{ts}",
                infra_standards=infra_conf,
                **pipe_conf,
            )
        except Exception as e:
            st.error(f"Could not load pipeline config: {e}")
            st.stop()
    _custom_rules_ex = _get_active_rules("ex")
    if _custom_rules_ex:
        rules_conf = _custom_rules_ex
    _start_run(pipe_conf, db_conf, rules_conf, infra_conf, pipeline_id, task)

# ---------------------------------------------------------------------------
# LIVE OUTPUT
# ---------------------------------------------------------------------------

if st.session_state.get("run_status") == "running":
    meta = st.session_state.pipeline_meta
    st.divider()

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pipeline",  meta["pipeline_id"])
    c2.metric("Cloud",     _CLOUD_FLAGS.get(meta["cloud"].lower(), meta["cloud"]))
    c3.metric("Database",  meta["db"].upper())
    c4.metric("Rules",     meta["rules"])

    st.divider()

    log_q: queue.Queue    = st.session_state.log_q
    state_q: queue.Queue  = st.session_state.state_q
    thread: threading.Thread = st.session_state.agent_thread

    # ── Execution metrics row (dynamic, updated every poll cycle) ─────────
    _exec_cols = st.columns(4)
    _time_ph   = _exec_cols[0].empty()
    _cycles_ph = _exec_cols[1].empty()
    _files_ph  = _exec_cols[2].empty()
    _cloud_ph  = _exec_cols[3].empty()

    # Local copies of visualization state — kept in sync with session_state
    _node_statuses  = dict(st.session_state.get("node_statuses",
                           {k: "pending" for k, _ in _PIPELINE_NODES}))
    _agent_events = list(st.session_state.get("agent_events", []))
    _run_counts = dict(st.session_state.get("run_counts", {}))
    _healing_cycles = st.session_state.get("healing_cycles", 0)
    _run_start_time = st.session_state.get("run_start_time", time.time())
    _last_edge = None  # (from, to) of the transition just taken — highlighted in the graph
    _graph_sig = None  # last-rendered graph state — re-render the animated SVG only when it changes

    st.divider()

    with st.status("Agent is running…", expanded=True) as status_box:
        # Anchor marker — the CSS scopes the sticky tab bar to ONLY these run tabs
        # (the top-level navigation tabs precede this anchor → untouched).
        st.markdown('<span class="sticky-tabs-anchor"></span>', unsafe_allow_html=True)
        _tab_log, _tab_graph = st.tabs(["📋 Activity log", "🕸️ Agent graph"])
        with _tab_log:
            # Timeline is the MAIN view — expandable, LangSmith-style (tool calls + inputs +
            # outputs per step). The raw routing/log stream is demoted to a collapsed expander.
            st.markdown(
                '<p style="color:#475569;font-size:0.72rem;text-transform:uppercase;'
                'letter-spacing:0.08em;margin:0 0 0.3rem;">🧭 Agent activity timeline '
                '<span style="text-transform:none;color:#334155;">— click a step to see its '
                'tools, inputs &amp; outputs</span></p>',
                unsafe_allow_html=True,
            )
            _agent_log_ph = st.empty()
            files_area = st.empty()
            with st.expander("🔎 Raw log (routing + runtime output)", expanded=False):
                log_area = st.empty()
        with _tab_graph:
            st.caption(
                "Hub-and-spoke: the **Supervisor** routes to Architect / Infra / Medic and each "
                "returns to it. The active node + the edge just taken are highlighted. The "
                "Supervisor is the **mission-outcome** node — it turns green only when the "
                "deployment is verified end-to-end, red on failure (amber if merely unverified)."
            )
            graph_ph = st.empty()
            st.markdown(_GRAPH_LEGEND_HTML, unsafe_allow_html=True)

        log_lines: list[str] = []
        written_files: list[str] = []

        while thread.is_alive() or not log_q.empty() or not state_q.empty():
            while not log_q.empty():
                _, msg = log_q.get_nowait()
                log_lines.append(msg)
            while not state_q.empty():
                upd = state_q.get_nowait()
                if "written_files" in upd:
                    written_files = upd["written_files"]
                    st.session_state.written_files = written_files
                if upd.get("status") in ("DONE", "ERROR"):
                    st.session_state.final_status = upd
                # Node graph updates
                if "node_update" in upd:
                    nu = upd["node_update"]
                    completed = nu.get("completed", "")
                    next_active = nu.get("next_active")
                    if completed in _node_statuses:
                        # The Supervisor is the hub/router, not a task — being visited is not
                        # "done". It gets its outcome colour (green/red/amber) only at the end.
                        _node_statuses[completed] = (
                            "hub" if completed == "supervisor" else "completed")
                        _run_counts[completed] = _run_counts.get(completed, 0) + 1
                        st.session_state.run_counts = dict(_run_counts)
                    if next_active and next_active in _node_statuses:
                        _node_statuses[next_active] = "active"
                    if completed and next_active:
                        _last_edge = (completed, next_active)
                    st.session_state.node_statuses = dict(_node_statuses)
                # Agent events — append in execution order (chronological timeline)
                if "agent_event" in upd:
                    _agent_events.append(upd["agent_event"])
                    st.session_state.agent_events = list(_agent_events)
                # Healing cycle counter
                if "healing_cycle" in upd:
                    _healing_cycles += upd["healing_cycle"]
                    st.session_state.healing_cycles = _healing_cycles

            # ── Refresh UI placeholders ───────────────────────────────────
            log_area.code("\n".join(log_lines[-80:]), language=None)
            _agent_log_ph.markdown(
                _render_agent_log_html(_agent_events), unsafe_allow_html=True
            )
            if written_files:
                files_area.markdown(
                    "<p style='color:#64748b; font-size:0.85rem;'>📁 "
                    + " &nbsp;·&nbsp; ".join(f"<code>{f}</code>" for f in written_files)
                    + "</p>",
                    unsafe_allow_html=True,
                )

            # Agent graph (center tab) — animated hub. Re-render ONLY on state change so the CSS
            # animation isn't restarted every poll (the iframe persists between changes).
            _gsig = (tuple(sorted(_node_statuses.items())),
                     tuple(sorted(_run_counts.items())), _last_edge)
            if _gsig != _graph_sig:
                _graph_sig = _gsig
                with graph_ph.container():
                    components.html(
                        _render_agent_svg(_node_statuses, _last_edge, _run_counts), height=350)

            # Execution metrics
            elapsed = int(time.time() - _run_start_time)
            _time_ph.metric("⏱️ Elapsed", f"{elapsed}s")
            _cycles_ph.metric("🔄 Healing Cycles", _healing_cycles)
            _files_ph.metric("📁 Files Generated", len(written_files))
            _cloud_ph.metric(
                "☁️ Cloud",
                _CLOUD_FLAGS.get(meta["cloud"].lower(), meta["cloud"]),
            )

            time.sleep(0.25)

        # Final drain
        while not log_q.empty():
            _, msg = log_q.get_nowait()
            log_lines.append(msg)
        while not state_q.empty():
            upd = state_q.get_nowait()
            if upd.get("status"):
                st.session_state.final_status = upd

        log_area.code("\n".join(log_lines), language=None)
        _agent_log_ph.markdown(
            _render_agent_log_html(_agent_events), unsafe_allow_html=True
        )

        final = st.session_state.get("final_status", {})
        if final.get("status") == "DONE":
            status_box.update(label="✅ Deployment complete!", state="complete")
            st.session_state.run_status = "done"
            # Mark any still-pending/active spoke as completed on clean exit; the Supervisor
            # turns green ONLY here (mission verified) — a green Supervisor means real success.
            for k in _node_statuses:
                if _node_statuses[k] in ("active", "pending", "hub"):
                    _node_statuses[k] = "completed"
        elif final.get("status") == "ERROR":
            status_box.update(label="❌ Deployment failed", state="error")
            st.session_state.run_status = "error"
            # The Supervisor is the MISSION-OUTCOME indicator: red on failure, amber when the
            # outcome is merely unverified/escalated (not a hard failure). Completed spokes STAY
            # green (option A) — they did their part; the failure is at the mission level.
            _ms = final.get("mission_status", "")
            _node_statuses["supervisor"] = (
                "warn" if _ms in ("escalated", "ci_unverified") else "failed")
            for k in _node_statuses:
                if _node_statuses[k] == "active":
                    _node_statuses[k] = "failed"

        # Final graph refresh. Clear the active edge so no transition arrow keeps animating on
        # the finished graph — the run is over, nothing is "in transit" any more.
        _last_edge = None
        with graph_ph.container():
            components.html(
                _render_agent_svg(_node_statuses, _last_edge, _run_counts), height=350)
        elapsed = int(time.time() - _run_start_time)
        _time_ph.metric("⏱️ Elapsed", f"{elapsed}s")
        _cycles_ph.metric("🔄 Healing Cycles", _healing_cycles)
        _files_ph.metric("📁 Files Generated", len(written_files))
        _cloud_ph.metric(
            "☁️ Cloud",
            _CLOUD_FLAGS.get(meta["cloud"].lower(), meta["cloud"]),
        )

    # Post-run
    if st.session_state.run_status == "done":
        st.markdown("""
        <div style="
            background: rgba(74,222,128,0.08);
            border: 1px solid rgba(74,222,128,0.3);
            border-radius: 14px;
            padding: 1.2rem 1.5rem;
            margin-top: 1rem;
        ">
            <p style="margin:0; color:#4ade80; font-weight:700; font-size:1.05rem;">
                🎉 Pipeline deployed successfully!
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.written_files:
            st.markdown("### 📁 Generated Files")
            for f in st.session_state.written_files:
                st.markdown(f"- `{f}`")

    elif st.session_state.run_status == "error":
        err = st.session_state.get("final_status", {}).get("error", "See log above.")
        st.error(f"Deployment failed: {err}")

    st.divider()
    if st.button("🔄 Run another pipeline", type="secondary"):
        for key in ["run_status", "log_q", "state_q", "agent_thread",
                    "pipeline_meta", "written_files", "final_status"]:
            st.session_state.pop(key, None)
        # Also drop any leftover NL wizard state so the next pipeline starts from a clean
        # slate (belt-and-suspenders alongside the step-2 stale-rules guard).
        for _nk in [k for k in st.session_state if k.startswith("nl_")]:
            del st.session_state[_nk]
        st.rerun()
