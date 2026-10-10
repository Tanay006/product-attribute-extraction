"""Page 2 - Model Evaluation + Audit. Reads ONLY real files produced by evaluate.py / audit.py."""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import results
import ui
from common import ATTRIBUTES, OUT_DIR
from pipeline import dataset_summary, get_model, load_split

nlp, status = get_model()
stamp = results.audit_stamp()
audit = results.load_audit(stamp)
spacy_eval = results.load_spacy_eval(stamp)

fmt = lambda v: f"{float(v):.3f}"                       # noqa: E731
md_safe = lambda s: str(s).replace("$", "\\$")           # noqa: E731


def pills():
    if audit is None:
        return ui.pill("No results yet", "bad")
    return ui.pill("Audit results loaded", "ok") + ui.pill(audit.model_name, "info")


ui.header("Model Evaluation + Audit",
          "Test-set metrics, subgroup comparison, error analysis, model limitations and audit findings, "
          "all read from the real evaluation outputs of this project.", pills())


# ----------------------------------------------------------------------------- (re)generate results
def generate_block(expanded=False):
    with st.expander("Generate / refresh the evaluation outputs from the current model", expanded=expanded):
        if not status.ok:
            st.info("The trained model (`" + status.path + "/`) is needed first. " + status.message)
            return
        st.write("Runs the project's own scripts in order: `evaluate.py` → `predict_spacy.py` → `audit.py`. "
                 "This scores the model on `data/test.json` and rewrites the files in `outputs/` and `audit_outputs/`. On the machine used for testing it took about 30 seconds.")
        if ui.button("Run evaluation + audit now", type="primary", key="gen_btn"):
            with st.status("Running the evaluation pipeline...", expanded=True) as stt:
                ok = results.run_pipeline(stt)
                stt.update(label="Finished" if ok else "Failed", state="complete" if ok else "error")
            if ok:
                st.cache_data.clear()
                st.rerun()


# ----------------------------------------------------------------------------- spaCy-scorer results (evaluate.py)
def spacy_scorer_block():
    m = spacy_eval.get("metrics")
    if not m:
        return
    st.markdown("#### spaCy's built-in scorer (from evaluate.py)")
    rows = [{"scope": "Overall", **m["overall"]}] + [{"scope": k, **v} for k, v in m.get("per_attribute", {}).items()]
    ui.table(pd.DataFrame(rows).rename(columns=str.lower))
    st.caption("The two scorers align spans to different tokenisations (spaCy tokens here, the dataset's own tokens in "
               "`audit.py`), so small differences between them are expected.")


def training_block():
    hist = spacy_eval.get("history")
    if hist is None or hist.empty:
        st.info("No training history found. `train.py` writes `outputs/training_history.json` when it finishes.")
        return
    best = hist.loc[hist["val_f1"].idxmax()]
    ui.kpi_cards([("Epochs run", str(len(hist)), "until early stopping / max epochs"),
                  ("Best validation F1", fmt(best["val_f1"]), f"epoch {int(best['epoch'])}"),
                  ("Final training loss", f"{hist['loss'].iloc[-1]:,.1f}", f"epoch {int(hist['epoch'].iloc[-1])}")])
    fig = go.Figure()
    for col, name in [("val_precision", "Precision"), ("val_recall", "Recall"), ("val_f1", "F1")]:
        fig.add_trace(go.Scatter(x=hist["epoch"], y=hist[col], mode="lines+markers", name=name,
                                 line=dict(color=ui.PRF_COLORS[col.replace("val_", "")])))
    fig.update_layout(title="Validation scores per epoch (validation set, not the test set)", yaxis=dict(range=[0, 1.02], title="Score"),
                      xaxis=dict(title="Epoch", dtick=1))
    ui.plot(fig)
    fig2 = go.Figure(go.Scatter(x=hist["epoch"], y=hist["loss"], mode="lines+markers", line=dict(color="#8172B2")))
    fig2.update_layout(title="Training loss per epoch", xaxis=dict(title="Epoch", dtick=1), yaxis=dict(title="Loss"))
    ui.plot(fig2)
    ui.table(hist)


# ============================================================================= NO RESULTS YET
if audit is None:
    ui.callout("warn", "No evaluation results found yet",
               "This page only shows <b>real</b> numbers produced by running the model on the test set, and those files "
               "(<code>audit_outputs/metrics.json</code> and the CSV tables) do not exist yet. Nothing is simulated or pre-filled.")
    st.markdown("#### How to generate them")
    st.code("python train.py        # 1. trains the model into ./model (skip if you already have it)\n"
            "python evaluate.py     # 2. spaCy scorer  -> outputs/\n"
            "python predict_spacy.py --model model --test data/test.json --out predictions.json\n"
            "python audit.py --preds predictions.json --model-name \"spaCy NER\"   # 3. audit -> audit_outputs/",
            language="bash")
    generate_block(expanded=status.ok)
    if spacy_eval:
        st.markdown("---")
        st.markdown("`evaluate.py` results were found, so here they are while the audit is still missing:")
        spacy_scorer_block()
        training_block()
    st.stop()

# ============================================================================= RESULTS AVAILABLE
if audit.is_stand_in:
    ui.callout("error", "These results come from a stand-in model, not your trained spaCy model",
               f"The results file names the model as <b>{audit.model_name}</b>. Do not report these numbers. "
               "Delete the <code>audit_outputs/</code> folder and regenerate it from the real model "
               "(button below or <code>audit.py</code>).")
if status.ok and results.results_older_than_model(audit, status.mtime):
    ui.callout("warn", "The model is newer than these results",
               "The files in <code>model/</code> were modified after the audit was generated, so the numbers below may "
               "describe an older model. Regenerate them to be sure.")

n_test = len(load_split("test"))
st.caption(f"Model evaluated: **{audit.model_name}** · results generated {audit.generated:%d %b %Y, %H:%M} · "
           f"`data/test.json` currently has {n_test:,} records · all numbers are entity-level on the held-out test set.")
generate_block()

tabs = st.tabs(["Overview", "Subgroups", "Error analysis", "Audit & limitations", "Training", "Downloads"])
T = audit.tables


# ----------------------------------------------------------------------------- Overview
with tabs[0]:
    ov, at = T.get("overall"), T.get("attr")
    if ov is not None:
        row = lambda prefix: ov[ov["metric"].str.startswith(prefix)].iloc[0]       # noqa: E731
        micro, macro, lenient = row("Micro"), row("Macro"), row("Lenient")
        ui.kpi_cards([("Precision", fmt(micro.precision), "strict micro-average"),
                      ("Recall", fmt(micro.recall), "strict micro-average"),
                      ("F1 score", fmt(micro.f1), "strict micro-average"),
                      ("Macro F1", fmt(macro.f1), "mean over 4 attributes"),
                      ("Lenient F1", fmt(lenient.f1), "same label, overlapping span")])
        with st.expander("How are these metrics defined?"):
            st.markdown(
                "- **Strict (exact match):** a prediction counts only if the attribute label **and** the exact span match the gold annotation.\n"
                "- **Micro-average:** pools all entities from all attributes before computing precision / recall / F1.\n"
                "- **Macro-average:** computes each attribute's score separately, then averages them equally.\n"
                "- **Lenient:** a prediction counts if it has the same label and overlaps the gold span. The gap between strict and "
                "lenient shows how much error is only about span boundaries.")
        ui.table(ov.round(3))
    if at is not None:
        fig = go.Figure()
        for col, name in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]:
            fig.add_trace(go.Bar(x=[ui.pretty(a) for a in at["attribute"]], y=at[col], name=name,
                                 marker_color=ui.PRF_COLORS[col], text=at[col].round(2), textposition="outside", textfont=dict(size=11)))
        fig.update_layout(title="Per-attribute scores (test set, strict)", barmode="group",
                          yaxis=dict(range=[0, 1.12], title="Score"))
        ui.plot(fig)
        ui.table(at.round(3))
        weakest = at.sort_values("f1").iloc[0]
        strongest = at.sort_values("f1").iloc[-1]
        st.markdown(f"**Reading the chart:** {ui.pretty(strongest.attribute)} scores highest (F1 {fmt(strongest.f1)}) and "
                    f"{ui.pretty(weakest.attribute)} lowest (F1 {fmt(weakest.f1)}).")
    with st.expander("Dataset used (computed from the files in data/)"):
        ds = pd.DataFrame(dataset_summary())
        ui.table(ds.rename(columns={"split": "Split", "records": "Records", "short_texts": "Short texts", "long_texts": "Long texts"}))
        st.caption("Attribute columns count gold entity spans per split.")
    spacy_scorer_block()


# ----------------------------------------------------------------------------- Subgroups
def group_chart(df, title, horizontal=False):
    d = df.copy()
    small = d["small_sample"].astype(bool)
    colors = ["#c7ccd4" if s else "#4C72B0" for s in small]
    txt_colors = ["#374151" if s else "#ffffff" for s in small]
    labels = [f"{f:.2f}  (n={n}{', unreliable' if s else ''})" for f, n, s in zip(d.f1, d.n_records, small)]
    err_hi, err_lo = (d.f1_ci_high - d.f1).clip(lower=0), (d.f1 - d.f1_ci_low).clip(lower=0)
    if horizontal:
        fig = go.Figure(go.Bar(y=d.subgroup, x=d.f1, orientation="h", marker_color=colors, text=labels,
                               textposition="inside", insidetextanchor="start", textfont=dict(color=txt_colors),
                               error_x=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo)))
        fig.update_layout(xaxis=dict(range=[0, 1.05], title="F1 (95% bootstrap CI)"), yaxis=dict(autorange="reversed"),
                          height=max(300, 55 * len(d) + 90))
    else:
        fig = go.Figure(go.Bar(x=d.subgroup, y=d.f1, marker_color=colors, text=labels,
                               textposition="inside", insidetextanchor="start", textfont=dict(color=txt_colors),
                               error_y=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo)))
        fig.update_layout(yaxis=dict(range=[0, 1.05], title="F1 (95% bootstrap CI)"))
    fig.update_layout(title=title, showlegend=False)
    ui.plot(fig)


with tabs[1]:
    sub, ent = T.get("sub"), T.get("ent")
    if sub is None:
        st.info("`subgroup_results.csv` was not found in audit_outputs/.")
    else:
        st.markdown("Grey bars are subgroups with fewer than 30 test records: their scores are **unreliable** and should not be compared.")
        for gtype, title, horiz in [("length_group", "F1: short vs long descriptions", False),
                                    ("length_bin", "F1 by description length (words)", False),
                                    ("category", "F1 by product category", True)]:
            d = sub[sub["group_type"] == gtype]
            if len(d):
                group_chart(d, title, horiz)
        show = sub.copy()
        show["95% CI"] = show.apply(lambda r: f"{r.f1_ci_low:.3f} – {r.f1_ci_high:.3f}", axis=1)
        show["reliable"] = show["small_sample"].map({True: "no (small sample)", False: "yes"})
        ui.table(show[["group_type", "subgroup", "n_records", "n_entities", "precision", "recall", "f1", "95% CI", "reliable"]]
                 .round(3))
    if ent is not None:
        d = ent.copy()
        d["label"] = d["property"] + ": " + d["group"]
        fig = go.Figure(go.Bar(x=d["label"], y=d["recall"], marker_color="#55A868",
                               text=[f"{r:.2f} (n={n})" for r, n in zip(d.recall, d.n_entities)], textposition="outside"))
        fig.update_layout(title="Entity-level recall by entity property", yaxis=dict(range=[0, 1.15], title="Recall"), showlegend=False)
        ui.plot(fig)
        st.caption("'Seen in training' = the exact phrase was tagged with that attribute somewhere in the training data.")


# ----------------------------------------------------------------------------- Error analysis
with tabs[2]:
    err = T.get("err")
    if err is None or err.empty:
        if err is None:
            st.info("`errors.csv` was not found in audit_outputs/.")
        else:
            st.success("No errors were recorded in errors.csv.")
    else:
        meaning = {"missed": "gold attribute not predicted at all", "boundary_error": "right label but span too short / long",
                   "wrong_label": "span found but given the wrong attribute", "false_alarm": "predicted attribute with no gold span nearby"}
        counts = err["type"].value_counts()
        ui.kpi_cards([("Total errors", f"{len(err):,}", "all types")] +
                     [(k.replace("_", " ").title(), f"{int(counts.get(k, 0)):,}", v) for k, v in meaning.items()])
        order = list(meaning)
        ct = err.groupby(["attribute", "type"]).size().unstack(fill_value=0).reindex(ATTRIBUTES).fillna(0)
        fig = go.Figure()
        palette = {"missed": "#C44E52", "boundary_error": "#DD8452", "wrong_label": "#8172B2", "false_alarm": "#937860"}
        for k in order:
            if k in ct:
                fig.add_trace(go.Bar(y=[ui.pretty(a) for a in ct.index], x=ct[k], name=k.replace("_", " "), orientation="h",
                                     marker_color=palette[k]))
        fig.update_layout(title="Error types per attribute", barmode="stack", xaxis=dict(title="Number of errors"),
                          yaxis=dict(autorange="reversed"), legend=dict(traceorder="normal"))
        ui.plot(fig)

        if "plausible_unlabeled" in err:
            fa = err[err["type"] == "false_alarm"]
            if len(fa):
                share = fa["plausible_unlabeled"].astype(str).str.lower().eq("true").mean()
                ui.callout("info", f"{share * 100:.0f}% of false alarms are phrases the annotators tag elsewhere in training",
                           "Many 'false alarms' may therefore be genuine attribute mentions missing from the gold labels. "
                           "See the label-quality check under Audit &amp; limitations.")

        st.markdown("#### Most frequent individual errors")
        top = (err.fillna({"gold": "", "predicted": ""}).groupby(["type", "attribute", "gold", "predicted"]).size()
               .reset_index(name="count").sort_values("count", ascending=False).head(15))
        ui.table(top)

        st.markdown("#### Browse all errors")
        f1, f2, f3 = st.columns(3)
        types = f1.multiselect("Error type", order, default=order)
        attrs = f2.multiselect("Attribute", ATTRIBUTES, default=ATTRIBUTES)
        groups = f3.multiselect("Text length", sorted(err["length_group"].dropna().unique()), default=sorted(err["length_group"].dropna().unique()))
        q = st.text_input("Search in text / gold / predicted", "")
        view = err[err["type"].isin(types) & err["attribute"].isin(attrs) & err["length_group"].isin(groups)]
        if q:
            ql = q.lower()
            view = view[view[["text", "gold", "predicted"]].fillna("").apply(lambda c: c.str.lower().str.contains(ql, regex=False)).any(axis=1)]
        st.caption(f"{len(view):,} of {len(err):,} errors shown (first 500 rendered).")
        cols = [c for c in ["type", "attribute", "gold", "predicted", "category", "length_group", "text"] if c in view]
        ui.table(view[cols].fillna("").head(500), column_config={"text": st.column_config.TextColumn("text", width="large")})


# ----------------------------------------------------------------------------- Audit & limitations
with tabs[3]:
    st.markdown("#### Audit findings")
    st.caption("Generated by `audit.py` from the numbers above; the wording is the project's own, so review it before submitting.")
    for f in audit.meta.get("findings", []):
        st.markdown("- " + md_safe(f))
    if not audit.meta.get("findings"):
        st.info("No findings found in metrics.json.")

    st.markdown("#### Model limitations")
    for l in audit.meta.get("limitations", []):
        st.markdown("- " + md_safe(l))
    if not audit.meta.get("limitations"):
        st.info("No limitations found in metrics.json.")

    lq = audit.meta.get("label_quality")
    if lq:
        st.markdown("#### Gold-label quality check")
        st.markdown("Phrases that annotators tagged at least 25 times in training are searched for in the test texts; "
                    "occurrences that are **not** tagged in the test gold labels point to possibly missing annotations.")
        ui.kpi_cards([(f"{g.title()} texts", f"{v['rate'] * 100:.1f}%", f"{v['untagged']:,} of {v['total']:,} occurrences untagged")
                      for g, v in lq.items()])
    miss = T.get("miss")
    if miss is not None and len(miss):
        with st.expander(f"Examples of possibly missing gold labels ({len(miss):,} listed)"):
            lab = st.multiselect("Attribute", sorted(miss["label"].unique()), default=sorted(miss["label"].unique()), key="miss_lab")
            ui.table(miss[miss["label"].isin(lab)].head(300), column_config={"text": st.column_config.TextColumn("text", width="large")})


# ----------------------------------------------------------------------------- Training
with tabs[4]:
    training_block()


# ----------------------------------------------------------------------------- Downloads
with tabs[5]:
    st.markdown("All files below were produced by the project's scripts and are served unchanged.")
    mimes = {"report": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
             "metrics": "application/json"}
    labels = {"report": "Audit report (Word)", "metrics": "metrics.json", "overall": "overall_results.csv",
              "attr": "attribute_results.csv", "sub": "subgroup_results.csv", "ent": "entity_property_results.csv",
              "err": "errors.csv", "miss": "possible_missing_gold_labels.csv"}
    cols = st.columns(2)
    for i, (k, path) in enumerate(audit.files.items()):
        with cols[i % 2]:
            with open(path, "rb") as fh:
                ui.download(labels[k], fh.read(), os.path.basename(path), mimes.get(k, "text/csv"), key=f"dl_{k}")
    for fig_name in ("fig_attr.png", "fig_groups.png", "fig_err.png"):
        pth = os.path.join(results.AUDIT_DIR, fig_name)
        if os.path.exists(pth):
            with st.expander(f"Static chart produced by audit.py: {fig_name}"):
                st.image(pth)
