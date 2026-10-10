"""Shared UI helpers: light-mode CSS, cards, entity highlighting, Plotly styling."""
import html

import streamlit as st

from common import ATTRIBUTES

# Colours for the 4 attributes (backgrounds taken from the project's original gui.py)
ATTR_BG = {"SLEEVE_STYLE": "#ffd6a5", "NECKLINE": "#caffbf", "LENGTH": "#9bf6ff", "PATTERN": "#ffc6ff"}
ATTR_ACCENT = {"SLEEVE_STYLE": "#f59e0b", "NECKLINE": "#22c55e", "LENGTH": "#06b6d4", "PATTERN": "#d946ef"}
PRF_COLORS = {"precision": "#4C72B0", "recall": "#DD8452", "f1": "#55A868"}   # same palette as audit.py


def pretty(label: str) -> str:
    return label.replace("_", " ").title()


CSS = """
<style>
:root { color-scheme: light; --bg:#f5f7fb; --card:#ffffff; --line:#e3e8ef; --ink:#1f2937;
        --muted:#6b7280; --brand:#2563eb; --brand-soft:#eaf1ff; }
html, body, .stApp, [data-testid="stAppViewContainer"] { background: var(--bg) !important; color: var(--ink); }
[data-testid="stHeader"] { background: rgba(255,255,255,.94); border-bottom: 1px solid var(--line); }
.block-container { max-width: 1180px; padding: 4.6rem 1.4rem 3rem; }   /* clears the fixed top bar */
h1, h2, h3 { letter-spacing: -.01em; color: #111827; }
h2 { font-size: 1.35rem !important; margin-top: 1.4rem !important; }
h3 { font-size: 1.1rem !important; }
h4 { font-size: 1.05rem !important; margin-top: 1rem !important; }

.hero { background: linear-gradient(135deg,#eaf1ff 0%,#f4f0ff 100%); border:1px solid #dbe5ff;
        border-radius:16px; padding:1.1rem 1.3rem; margin-bottom:1rem; }
.hero h1 { margin:0; font-size:1.65rem; line-height:1.25; }
.hero p { margin:.35rem 0 0; color:#4b5563; font-size:.95rem; }
.pill { display:inline-block; padding:.12rem .6rem; border-radius:999px; font-size:.74rem; font-weight:700;
        letter-spacing:.03em; margin-right:.35rem; border:1px solid; }
.pill.ok { background:#f0fdf4; color:#166534; border-color:#bbf7d0; }
.pill.bad { background:#fef2f2; color:#991b1b; border-color:#fecaca; }
.pill.info { background:#eff6ff; color:#1e40af; border-color:#bfdbfe; }

.card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem;
        box-shadow:0 1px 2px rgba(16,24,40,.04); }
.kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.75rem; margin:.25rem 0 1rem; }
.kpi { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:.8rem 1rem;
       box-shadow:0 1px 2px rgba(16,24,40,.04); }
.kpi .label { font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; font-weight:600; }
.kpi .value { font-size:1.65rem; font-weight:700; color:#111827; line-height:1.25; }
.kpi .sub { font-size:.76rem; color:var(--muted); }

.callout { border-radius:12px; padding:.8rem 1rem; border:1px solid; margin:.4rem 0 1rem; font-size:.94rem; }
.callout b.t { display:block; margin-bottom:.15rem; }
.callout.info { background:#eff6ff; border-color:#bfdbfe; color:#1e3a8a; }
.callout.warn { background:#fffbeb; border-color:#fde68a; color:#78350f; }
.callout.error { background:#fef2f2; border-color:#fecaca; color:#7f1d1d; }
.callout.ok { background:#f0fdf4; border-color:#bbf7d0; color:#14532d; }
.callout code { background:rgba(0,0,0,.06); padding:.05rem .3rem; border-radius:4px; }

.step { display:flex; align-items:center; gap:.6rem; margin:1.3rem 0 .5rem; }
.step .n { background:var(--brand); color:#fff; width:1.7rem; height:1.7rem; border-radius:50%;
           display:inline-flex; align-items:center; justify-content:center; font-weight:700; font-size:.85rem; flex:none; }
.step .t { font-size:1.15rem; font-weight:700; color:#111827; }

.hl-box { background:#fff; border:1px solid var(--line); border-radius:12px; padding:1rem 1.1rem;
          line-height:2.3; font-size:1.05rem; overflow-wrap:anywhere; white-space:pre-wrap; }
mark.ent { padding:.18rem .4rem; border-radius:7px; margin:0 1px; color:#111827; }
mark.ent small { font-size:.6rem; font-weight:800; letter-spacing:.05em; margin-left:.4rem; opacity:.7; text-transform:uppercase; }
.legend { margin:.5rem 0 0; font-size:.82rem; color:var(--muted); }
.legend span.sw { display:inline-block; padding:.05rem .5rem; border-radius:6px; margin:0 .35rem .25rem 0; color:#111827; font-weight:600; }

.attr-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:.75rem; }
.attr-card { background:var(--card); border:1px solid var(--line); border-left:6px solid; border-radius:12px; padding:.75rem .95rem; }
.attr-card .name { font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
.attr-card .vals { margin-top:.3rem; min-height:1.9rem; }
.chip { display:inline-block; padding:.15rem .65rem; border-radius:999px; margin:.15rem .3rem .15rem 0; font-weight:600; font-size:.92rem; color:#111827; }
.none { color:#9ca3af; font-style:italic; }

div[data-testid="stTabs"] button[role="tab"] { font-weight:600; }
div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
.stButton > button, .stDownloadButton > button { border-radius:10px; font-weight:600; }
.small-note { color:var(--muted); font-size:.82rem; }

.st-key-mobile_nav { display:none; margin:-.4rem 0 .6rem; }
@media (max-width: 900px) { .st-key-mobile_nav { display:block; } }
@media (max-width: 640px) {
  .block-container { padding: 4.2rem .7rem 2.5rem; }
  .hero { padding:.85rem .95rem; } .hero h1 { font-size:1.3rem; }
  .kpi .value { font-size:1.35rem; }
  .hl-box { font-size:.98rem; line-height:2.2; padding:.8rem; }
  h2 { font-size:1.2rem !important; }
}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def mobile_nav():
    """Visible page links on small screens (the top bar collapses behind a tiny icon on phones)."""
    try:
        with st.container(key="mobile_nav"):
            st.page_link("views/extraction_page.py", label="NLP Model + Attribute Extraction", icon=":material/manage_search:")
            st.page_link("views/evaluation_page.py", label="Model Evaluation + Audit", icon=":material/fact_check:")
    except Exception:       # e.g. page run outside the multipage app (tests)
        pass


def header(title: str, subtitle: str, pills: str = ""):
    mobile_nav()
    st.markdown(f'<div class="hero"><div>{pills}</div><h1>{html.escape(title)}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def callout(kind: str, title: str, body_html: str = ""):
    """kind: info | warn | error | ok.  body_html must already be safe HTML."""
    st.markdown(f'<div class="callout {kind}"><b class="t">{html.escape(title)}</b>{body_html}</div>',
                unsafe_allow_html=True)


def pill(text: str, kind: str = "info") -> str:
    return f'<span class="pill {kind}">{html.escape(text)}</span>'


def step(n: int, title: str):
    st.markdown(f'<div class="step"><span class="n">{n}</span><span class="t">{html.escape(title)}</span></div>',
                unsafe_allow_html=True)


def kpi_cards(items):
    """items: list of (label, value_str, sub_str)."""
    cards = "".join(
        f'<div class="kpi"><div class="label">{html.escape(l)}</div><div class="value">{html.escape(v)}</div>'
        f'<div class="sub">{html.escape(s)}</div></div>' for l, v, s in items)
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)


def highlight_html(text: str, ents) -> str:
    """Render text with entity spans highlighted. ents: dicts with start_char, end_char, attribute."""
    out, cur = [], 0
    for e in sorted(ents, key=lambda x: x["start_char"]):
        if e["start_char"] < cur:
            continue
        out.append(html.escape(text[cur:e["start_char"]]))
        bg = ATTR_BG.get(e["attribute"], "#e5e7eb")
        out.append(f'<mark class="ent" style="background:{bg}">{html.escape(text[e["start_char"]:e["end_char"]])}'
                   f'<small>{html.escape(e["attribute"])}</small></mark>')
        cur = e["end_char"]
    out.append(html.escape(text[cur:]))
    return f'<div class="hl-box">{"".join(out)}</div>'


def legend_html() -> str:
    sw = "".join(f'<span class="sw" style="background:{ATTR_BG[a]}">{pretty(a)}</span>' for a in ATTRIBUTES)
    return f'<div class="legend">{sw}</div>'


def attribute_cards(by_attr) -> str:
    cards = []
    for a in ATTRIBUTES:
        vals = by_attr.get(a, [])
        body = ("".join(f'<span class="chip" style="background:{ATTR_BG[a]}">{html.escape(v)}</span>' for v in vals)
                if vals else '<span class="none">not found</span>')
        cards.append(f'<div class="attr-card" style="border-left-color:{ATTR_ACCENT[a]}"><div class="name">{pretty(a)}</div>'
                     f'<div class="vals">{body}</div></div>')
    return f'<div class="attr-grid">{"".join(cards)}</div>'


# ---- Streamlit-version-tolerant wrappers (width="stretch" in new versions, use_container_width in old) ----
def table(df, **kw):
    cfg = dict(kw.pop("column_config", None) or {})
    for col in df.select_dtypes("float").columns:
        cfg.setdefault(col, st.column_config.NumberColumn(format="%.3f"))
    if cfg:
        kw["column_config"] = cfg
    try:
        return st.dataframe(df, width="stretch", hide_index=True, **kw)
    except Exception:
        return st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def plot(fig, **kw):
    fig.update_layout(template="plotly_white", font=dict(family="sans-serif", size=13, color="#1f2937"),
                      margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="white", plot_bgcolor="white",
                      legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0),
                      title=dict(font=dict(size=15), x=0.01))
    cfg = {"displayModeBar": False}
    try:
        return st.plotly_chart(fig, width="stretch", config=cfg, **kw)
    except Exception:
        return st.plotly_chart(fig, use_container_width=True, config=cfg, **kw)


def button(label, **kw):
    try:
        return st.button(label, width="stretch", **kw)
    except Exception:
        return st.button(label, use_container_width=True, **kw)


def download(label, data, file_name, mime, **kw):
    try:
        return st.download_button(label, data, file_name, mime, width="stretch", **kw)
    except Exception:
        return st.download_button(label, data, file_name, mime, use_container_width=True, **kw)
