"""Page 1 - NLP Model + Attribute Extraction (real spaCy NER model, no placeholder predictions)."""
import io
import json
import random

import pandas as pd
import streamlit as st

import ui
from common import ATTRIBUTES, MODEL_DIR
from explain import occlusion                      # the project's own occlusion explainability
from pipeline import (MAX_BATCH_ROWS, MAX_CHARS, compare_with_gold, extract, extract_batch, get_model,
                      load_split, preprocess)

DEFAULT_TEXT = "Women's black floral maxi dress with short sleeves and a V-neck"   # default used in the project's gui.py

nlp, status = get_model()

ui.header(
    "NLP Model + Attribute Extraction",
    "Type a clothing product description. The trained spaCy NER model extracts "
    "<b>sleeve style, neckline, length</b> and <b>pattern</b>.",
    ui.pill("Model loaded" if status.ok else "Model not found", "ok" if status.ok else "bad")
    + ui.pill("spaCy " + status.spacy_installed, "info"),
)

# ----------------------------------------------------------------------------- model-loading step
if not status.ok:
    ui.callout("error", "The trained model is not available yet",
               f"<div>{status.message}</div><div style='margin-top:.4rem'>This page runs the <b>real</b> model, so it "
               f"cannot show predictions until the model exists. Nothing is simulated.</div>")
    st.markdown("#### Set up the model (one-time)")
    st.code("pip install -r requirements.txt\n"
            "python train.py --quick     # optional 1-2 min sanity check (2,000 records)\n"
            "python train.py             # full training; the best model is saved to ./model", language="bash")
    st.markdown(f"Already have the trained model folder from your team? Copy it here as <code>{MODEL_DIR}/</code> "
                f"(next to <code>app.py</code>), or set the environment variable <code>NER_MODEL_DIR</code> to its path. "
                f"Then click the button below.", unsafe_allow_html=True)
    if ui.button("Re-check for the model", type="primary"):
        st.cache_resource.clear()
        st.rerun()
    st.stop()

# ----------------------------------------------------------------------------- state / callbacks
st.session_state.setdefault("input_text", DEFAULT_TEXT)
st.session_state.setdefault("gold_rec", None)
st.session_state.setdefault("grp", "any")


def _load_random_record():
    recs = load_split("test")
    if st.session_state["grp"] != "any":
        recs = [r for r in recs if r["length_group"] == st.session_state["grp"]]
    rec = random.choice(recs)
    st.session_state["input_text"] = rec["text"]
    st.session_state["gold_rec"] = rec


def _reset_default():
    st.session_state["input_text"] = DEFAULT_TEXT
    st.session_state["gold_rec"] = None


tab_single, tab_batch = st.tabs(["Single description", "Batch (many descriptions)"])

# ============================================================================= SINGLE
def render_single():
    st.text_area("Product description", key="input_text", height=130, max_chars=MAX_CHARS,
                 help="Press Ctrl+Enter, click outside the box, or press the button to run the pipeline.")
    c1, c2, c3, c4 = st.columns([1.1, 1.2, 1.1, 1])
    with c1:
        st.selectbox("Test-set length group", ["any", "short", "long"], key="grp",
                     help="'short' = up to 13 words, 'long' = 14+ words (the dataset's own length_group).")
    with c2:
        st.write("")
        ui.button("Load a random test record", on_click=_load_random_record)
    with c3:
        st.write("")
        ui.button("Reset example", on_click=_reset_default)
    with c4:
        st.write("")
        ui.button("Extract attributes", type="primary")

    raw = st.session_state["input_text"]
    cleaned, tok_rows = preprocess(raw, nlp)
    if not cleaned:
        ui.callout("info", "Enter a product description above to run the pipeline.")
        return

    # ---- 1. preprocessing
    ui.step(1, "Input & preprocessing")
    ui.kpi_cards([("Characters", f"{len(cleaned):,}", "after trimming"),
                  ("Words", f"{len(cleaned.split()):,}", "whitespace-separated"),
                  ("Model tokens", f"{len(tok_rows):,}", "spaCy English tokenizer")])
    with st.expander("View the tokens the model receives"):
        ui.table(pd.DataFrame(tok_rows))
    ui.callout("info", "What preprocessing is applied?",
               "The model was trained on the <b>raw</b> text split by spaCy's English tokenizer (see <code>common.py</code>), "
               "so the app does exactly that and only trims outer whitespace. It deliberately does <b>not</b> lowercase, "
               "stem or remove stop-words, because that would change the model's input compared with training.")

    # ---- 2. model prediction
    ents, by_attr = extract(nlp, cleaned)
    ui.step(2, "Model prediction")
    st.markdown(ui.highlight_html(cleaned, ents), unsafe_allow_html=True)
    st.markdown(ui.legend_html(), unsafe_allow_html=True)
    st.caption("spaCy's NER component does not output per-entity confidence scores, so none are shown.")

    # ---- 3. attribute extraction
    ui.step(3, "Extracted attributes")
    if not ents:
        ui.callout("warn", "No attributes were detected in this text.",
                   "The model only extracts sleeve style, neckline, length and pattern. Texts that mention none of these "
                   "(or use phrases it has not learned) return nothing.")
    st.markdown(ui.attribute_cards(by_attr), unsafe_allow_html=True)
    if ents:
        df = pd.DataFrame([{"Attribute": e["attribute"], "Extracted text": e["text"], "Start char": e["start_char"],
                            "End char": e["end_char"], "Token range": f'{e["start_token"]}-{e["end_token"] - 1}'}
                           for e in ents])
        st.write("")
        ui.table(df)
        payload = {"text": cleaned, "attributes": by_attr, "entities": ents}
        d1, d2 = st.columns(2)
        with d1:
            ui.download("Download JSON", json.dumps(payload, indent=2), "attributes.json", "application/json")
        with d2:
            ui.download("Download CSV", df.to_csv(index=False), "attributes.csv", "text/csv")

    # ---- 4. comparison with gold labels (only for an untouched test record)
    gold = st.session_state.get("gold_rec")
    if gold and gold["text"] == raw:
        ui.step(4, "Comparison with the dataset's gold annotation")
        rows, n_hit, n_gold, n_pred = compare_with_gold(gold, ents)
        ui.kpi_cards([("Category", gold["category"], f'length group: {gold["length_group"]}'),
                      ("Gold spans matched", f"{n_hit} / {n_gold}", "exact span + attribute"),
                      ("Model spans", str(n_pred), "predicted for this record")])
        if rows:
            ui.table(pd.DataFrame(rows))
        st.caption("Gold labels in this dataset are known to be incomplete in some records (see the audit page), so a "
                   "'Not in gold labels' row is not automatically a model mistake.")

    # ---- 5. explainability
    ui.step(5 if (gold and gold["text"] == raw) else 4, "Explainability: which words did the model rely on?")
    with st.expander("Word-removal (occlusion) analysis", expanded=False):
        words = cleaned.split()
        st.write("Removes one word at a time and checks which extracted attributes disappear "
                 "(the project's own `explain.py`). A word whose removal loses an attribute is one the model relied on.")
        if len(words) > 60:
            st.info(f"This text has {len(words)} words; the analysis runs one model call per word, so it is limited to "
                    "texts of 60 words or fewer.")
        elif ui.button("Run occlusion analysis"):
            with st.spinner("Re-running the model with each word removed..."):
                base, res = occlusion(nlp, cleaned)
            impact = [{"Word": w, "Removing it loses": ", ".join(f"{t} [{l}]" for t, l in sorted(lost))}
                      for w, lost in res if lost]
            if impact:
                ui.table(pd.DataFrame(impact))
            else:
                st.info("Removing any single word did not change the extracted attributes for this text.")

with tab_single:
    render_single()

# ============================================================================= BATCH
with tab_batch:
    st.markdown("Run the same pipeline over many descriptions at once. Paste one description per line, "
                "or upload a CSV / TXT file.")
    up = st.file_uploader("Upload CSV or TXT", type=["csv", "txt"], help="CSV: pick the text column below. TXT: one description per line.")
    pasted = st.text_area("...or paste descriptions (one per line)", height=140, key="batch_text")

    texts = []
    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                df_in = pd.read_csv(up)
                col = st.selectbox("Column containing the descriptions", list(df_in.columns))
                texts = df_in[col].dropna().astype(str).tolist()
            else:
                texts = [l.strip() for l in io.StringIO(up.getvalue().decode("utf-8", errors="replace")) if l.strip()]
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the file: {e}")
    elif pasted.strip():
        texts = [l.strip() for l in pasted.splitlines() if l.strip()]

    if texts:
        skipped = [t for t in texts if len(t) > MAX_CHARS]
        texts = [t for t in texts if len(t) <= MAX_CHARS]
        if len(texts) > MAX_BATCH_ROWS:
            st.warning(f"Only the first {MAX_BATCH_ROWS:,} descriptions are processed.")
            texts = texts[:MAX_BATCH_ROWS]
        if skipped:
            st.warning(f"{len(skipped)} description(s) longer than {MAX_CHARS:,} characters were skipped.")
        if texts and ui.button("Extract attributes for all", type="primary", key="run_batch"):
            with st.spinner(f"Running the model on {len(texts):,} descriptions..."):
                res = extract_batch(nlp, [t.strip() for t in texts])
            out = pd.DataFrame({"description": texts})
            for a in ATTRIBUTES:
                out[a] = [" | ".join(r[a]) for r in res]
            out["attributes_found"] = [sum(bool(r[a]) for a in ATTRIBUTES) for r in res]
            st.session_state["batch_out"] = out

    out = st.session_state.get("batch_out")
    if out is not None:
        st.markdown("#### Results")
        ui.kpi_cards([("Descriptions", f"{len(out):,}", "processed")] +
                     [(ui.pretty(a), f"{(out[a] != '').mean() * 100:.0f}%", "texts with this attribute") for a in ATTRIBUTES])
        ui.table(out)
        ui.download("Download results (CSV)", out.to_csv(index=False), "batch_attributes.csv", "text/csv")
