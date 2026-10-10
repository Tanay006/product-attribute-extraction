"""Loads the REAL evaluation outputs written by evaluate.py (outputs/) and audit.py (audit_outputs/).
Nothing here computes or invents metrics - if a file is missing the page says so.
"""
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import streamlit as st

from common import MODEL_DIR, OUT_DIR

AUDIT_DIR = "audit_outputs"
STAND_IN_WORDS = ("stand-in", "standin", "surrogate", "preview")

AUDIT_FILES = {
    "metrics": "metrics.json", "overall": "overall_results.csv", "attr": "attribute_results.csv",
    "sub": "subgroup_results.csv", "ent": "entity_property_results.csv", "err": "errors.csv",
    "miss": "possible_missing_gold_labels.csv", "report": "audit_report.docx",
}


def _p(name):
    return os.path.join(AUDIT_DIR, AUDIT_FILES[name])


def audit_stamp():
    """Cache key: modification times of every audit file that exists."""
    return tuple(os.path.getmtime(_p(k)) if os.path.exists(_p(k)) else 0 for k in AUDIT_FILES) + (
        os.path.getmtime(os.path.join(OUT_DIR, "metrics.json")) if os.path.exists(os.path.join(OUT_DIR, "metrics.json")) else 0,
        os.path.getmtime(os.path.join(OUT_DIR, "training_history.json")) if os.path.exists(os.path.join(OUT_DIR, "training_history.json")) else 0,
    )


@dataclass
class Audit:
    meta: dict
    model_name: str
    generated: datetime
    is_stand_in: bool
    tables: dict = field(default_factory=dict)      # name -> DataFrame (only those that exist)
    files: dict = field(default_factory=dict)       # name -> path (only those that exist)


def audit_available() -> bool:
    return os.path.exists(_p("metrics"))


@st.cache_data(show_spinner=False)
def load_audit(stamp):
    if not audit_available():
        return None
    meta = json.load(open(_p("metrics"), encoding="utf-8"))
    name = str(meta.get("model", "unknown model"))
    tables = {k: pd.read_csv(_p(k)) for k in ("overall", "attr", "sub", "ent", "err", "miss")
              if os.path.exists(_p(k)) and os.path.getsize(_p(k)) > 2}
    files = {k: _p(k) for k in AUDIT_FILES if os.path.exists(_p(k))}
    return Audit(meta, name, datetime.fromtimestamp(os.path.getmtime(_p("metrics"))),
                 any(w in name.lower() for w in STAND_IN_WORDS), tables, files)


@st.cache_data(show_spinner=False)
def load_spacy_eval(stamp):
    """Outputs of evaluate.py / train.py (spaCy's own scorer + training curve)."""
    out = {}
    m = os.path.join(OUT_DIR, "metrics.json")
    h = os.path.join(OUT_DIR, "training_history.json")
    if os.path.exists(m):
        out["metrics"] = json.load(open(m, encoding="utf-8"))
    if os.path.exists(h):
        out["history"] = pd.DataFrame(json.load(open(h, encoding="utf-8")))
    return out


def results_older_than_model(audit: Audit, model_mtime: float) -> bool:
    return bool(model_mtime) and model_mtime > audit.generated.timestamp()


# ----------------------------------------------------------------------------- run the project scripts
def pipeline_commands(model_dir: str = MODEL_DIR):
    py = sys.executable
    return [
        ("evaluate.py  (spaCy scorer -> outputs/)", [py, "evaluate.py"]),
        ("predict_spacy.py  (token-level predictions)", [py, "predict_spacy.py", "--model", model_dir,
                                                         "--test", "data/test.json", "--out", "predictions.json"]),
        ("audit.py  (audit tables, charts, report -> audit_outputs/)",
         [py, "audit.py", "--preds", "predictions.json", "--train", "data/train.json", "--test", "data/test.json",
          "--model-name", f"spaCy NER ({model_dir})"]),
    ]


def run_pipeline(status_container, model_dir: str = MODEL_DIR) -> bool:
    """Runs the project's own scripts in order. Returns True if all succeeded."""
    env = {**os.environ, "NER_MODEL_DIR": model_dir, "PYTHONIOENCODING": "utf-8"}
    for label, cmd in pipeline_commands(model_dir):
        status_container.write(f"Running `{label}` ...")
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            status_container.error(f"`{label}` failed (exit code {proc.returncode}).")
            status_container.code((proc.stdout[-2000:] + "\n" + proc.stderr[-3000:]).strip(), language="text")
            return False
    return True
