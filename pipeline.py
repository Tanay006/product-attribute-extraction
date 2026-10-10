"""Inference side of the app: model loading/validation, preprocessing, attribute extraction.

Preprocessing mirrors training exactly (see common.py / train.py): the model was trained on the RAW
product text, tokenised by spaCy's English tokenizer (nlp.make_doc). No lower-casing, stemming or
stop-word removal was used in training, so none is applied here - doing so would change the model's input.
The only step the app adds (as the project's own gui.py did) is stripping leading/trailing whitespace.
"""
import json
import os
from dataclasses import dataclass, field

import streamlit as st
import spacy

from common import ATTRIBUTES, DATA_DIR, MODEL_DIR, bio_to_char_spans, load_jsonl

MAX_CHARS = 5000        # longest record in the dataset is ~2,000 characters
MAX_BATCH_ROWS = 2000


# ----------------------------------------------------------------------------- model loading
@dataclass
class ModelStatus:
    ok: bool
    path: str
    message: str = ""
    labels: tuple = ()
    spacy_required: str = ""
    spacy_installed: str = spacy.__version__
    mtime: float = 0.0
    pipe_names: list = field(default_factory=list)


def model_mtime(path: str = MODEL_DIR) -> float:
    newest = 0.0
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                newest = max(newest, os.path.getmtime(os.path.join(root, f)))
    return newest


@st.cache_resource(show_spinner="Loading NER model...")
def _load(path: str, mtime: float):
    return spacy.load(path)


def get_model():
    """Return (nlp | None, ModelStatus). Never raises - problems are reported in the status."""
    path = MODEL_DIR
    if not os.path.isdir(path):
        return None, ModelStatus(False, path, f"The model folder '{path}/' was not found next to app.py.")
    if not (os.path.exists(os.path.join(path, "meta.json")) and os.path.exists(os.path.join(path, "config.cfg"))):
        return None, ModelStatus(False, path, f"'{path}/' exists but is not a saved spaCy pipeline "
                                              "(meta.json / config.cfg missing).")
    mt = model_mtime(path)
    required = ""
    try:
        required = json.load(open(os.path.join(path, "meta.json"), encoding="utf-8")).get("spacy_version", "")
    except Exception:
        pass
    try:
        nlp = _load(path, mt)
    except Exception as e:  # noqa: BLE001 - surface any loading problem to the user
        return None, ModelStatus(False, path, f"spaCy could not load '{path}/': {e}", spacy_required=required, mtime=mt)
    if "ner" not in nlp.pipe_names:
        return None, ModelStatus(False, path, "The loaded pipeline has no 'ner' component.", mtime=mt,
                                 pipe_names=list(nlp.pipe_names))
    labels = tuple(nlp.get_pipe("ner").labels)
    missing = [a for a in ATTRIBUTES if a not in labels]
    if missing:
        return None, ModelStatus(False, path, f"The model does not know these attributes: {', '.join(missing)}. "
                                              "It was not trained with this project's labels.",
                                 labels=labels, mtime=mt, pipe_names=list(nlp.pipe_names))
    return nlp, ModelStatus(True, path, "Model loaded.", labels, required, spacy.__version__, mt, list(nlp.pipe_names))


# ----------------------------------------------------------------------------- preprocessing
def preprocess(text: str, nlp):
    """Returns (cleaned_text, token_rows). Tokens are exactly what the NER component sees."""
    cleaned = text.strip()
    doc = nlp.make_doc(cleaned)
    rows = [{"#": t.i, "token": t.text, "start_char": t.idx, "end_char": t.idx + len(t.text)}
            for t in doc if not t.is_space]
    return cleaned, rows


# ----------------------------------------------------------------------------- extraction
def _unique(values):
    seen, out = set(), []
    for v in values:
        k = v.lower()
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


def _ents(doc):
    return [dict(attribute=e.label_, text=e.text, start_char=e.start_char, end_char=e.end_char,
                 start_token=e.start, end_token=e.end) for e in doc.ents]


def group_by_attribute(ents):
    by = {a: [] for a in ATTRIBUTES}
    for e in ents:
        by.setdefault(e["attribute"], []).append(e["text"])
    return {a: _unique(v) for a, v in by.items()}


def extract(nlp, text: str):
    """Run the real model. Returns (entities, by_attribute)."""
    ents = _ents(nlp(text))
    return ents, group_by_attribute(ents)


def extract_batch(nlp, texts):
    out = []
    for doc in nlp.pipe(texts, batch_size=64):
        out.append(group_by_attribute(_ents(doc)))
    return out


# ----------------------------------------------------------------------------- gold comparison
def compare_with_gold(record, ents):
    """Compare predictions with the dataset's gold spans for a loaded test record (exact span + label)."""
    text = record["text"]
    shift = len(text) - len(text.lstrip())          # input is stripped, so shift gold offsets
    gold = {(s - shift, e - shift, lab) for s, e, lab in bio_to_char_spans(record)}
    pred = {(e["start_char"], e["end_char"], e["attribute"]) for e in ents}
    cleaned = text.strip()
    rows = []
    for s, e, lab in sorted(gold):
        rows.append({"source": "Gold", "attribute": lab, "text": cleaned[s:e],
                     "result": "Matched by model" if (s, e, lab) in pred else "Not matched exactly"})
    for s, e, lab in sorted(pred - gold):
        rows.append({"source": "Model", "attribute": lab, "text": cleaned[s:e], "result": "Not in gold labels"})
    n_hit = len(gold & pred)
    return rows, n_hit, len(gold), len(pred)


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_split(name: str):
    return load_jsonl(os.path.join(DATA_DIR, f"{name}.json"))


def _bio_spans(tags):
    """Same span logic as audit.py: BIO tags -> [(label, start_tok, end_tok_exclusive)]."""
    spans, cur = [], None
    for i, t in enumerate(tags):
        if t.startswith("B-") or (t.startswith("I-") and (cur is None or cur[0] != t[2:])):
            if cur:
                spans.append(tuple(cur))
            cur = [t[2:], i, i + 1]
        elif t.startswith("I-"):
            cur[2] = i + 1
        else:
            if cur:
                spans.append(tuple(cur))
            cur = None
    if cur:
        spans.append(tuple(cur))
    return spans


@st.cache_data(show_spinner=False)
def dataset_summary():
    rows = []
    for name in ("train", "val", "test"):
        recs = load_split(name)
        row = {"split": name, "records": len(recs)}
        counts = {a: 0 for a in ATTRIBUTES}
        for r in recs:
            for lab, _, _ in _bio_spans(r["labels"]):
                counts[lab] = counts.get(lab, 0) + 1
        row.update(counts)
        row["short_texts"] = sum(r["length_group"] == "short" for r in recs)
        row["long_texts"] = sum(r["length_group"] == "long" for r in recs)
        rows.append(row)
    return rows
