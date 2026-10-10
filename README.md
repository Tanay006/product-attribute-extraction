# CA3 - Product Attribute Extraction (spaCy NER) - Streamlit app

A two-page web app on top of the project's existing spaCy NER model, dataset and scripts.
It extracts **SLEEVE_STYLE, NECKLINE, LENGTH and PATTERN** from clothing product descriptions and shows the
model's real evaluation and audit results.

> **No fake numbers.** The app contains no dummy predictions and no pre-filled metrics. Page 1 runs the real
> trained model; Page 2 reads the real files written by `evaluate.py` and `audit.py`. If those do not exist yet,
> each page says so and tells you what to run.

## Pages

| Page | What it does |
|---|---|
| **NLP Model + Attribute Extraction** | Text input -> preprocessing (shows the exact tokens the model sees) -> real model prediction (highlighted entities) -> structured attribute cards, span table, JSON/CSV download. Extras: load a random record from the test set and compare with its gold annotation, word-removal (occlusion) explainability, and a batch mode (paste lines or upload CSV/TXT). |
| **Model Evaluation + Audit** | Overview (strict micro P/R/F1, macro F1, lenient F1, per-attribute chart), subgroups (short/long, length bins, product category, with 95% bootstrap CIs and small-sample flags), error analysis (types, top recurring errors, filterable browser), audit findings and model limitations, gold-label quality check, training curves, and downloads of every output file. |

Light mode only, responsive for mobile, tablet and laptop (on small screens visible page links are added because
the top bar collapses behind an icon).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Get the trained model into ./model  (pick one)
python train.py --quick     # ~2,000 records, 3 epochs: only a sanity check
python train.py             # full training; best model by validation F1 is saved to ./model
#    ...or copy your team's existing model folder here as ./model

# 2) Run the app
streamlit run app.py
```

### The model-loading step
On start-up the app looks for `model/` next to `app.py` and checks that it is a saved spaCy pipeline with an
`ner` component that knows all four attributes. If it is missing or invalid, Page 1 explains why and shows what to
do (nothing is simulated). To use a different location set `NER_MODEL_DIR`:

```bash
NER_MODEL_DIR=/path/to/model streamlit run app.py      # Windows (PowerShell): $env:NER_MODEL_DIR="C:\path\to\model"
```

### Generating the evaluation results
Either click **"Run evaluation + audit now"** on Page 2 (it runs the three scripts below), or run them yourself:

```bash
python evaluate.py                                                    # -> outputs/   (spaCy scorer, errors, subgroups)
python predict_spacy.py --model model --test data/test.json --out predictions.json
python audit.py --preds predictions.json --model-name "spaCy NER"     # -> audit_outputs/ (tables, charts, Word report)
```
Then refresh Page 2. On the machine used for development the three steps took about 30 seconds in total.

## Built-in safeguards
- **Stand-in results guard:** if `audit_outputs/metrics.json` names a stand-in / surrogate / preview model, Page 2 shows a red banner. Do **not** copy `audit_outputs_preview/` from the audit hand-off into this folder.
- **Stale results guard:** if the files in `model/` are newer than the audit results, Page 2 warns you to regenerate them.

## Preprocessing - what and why
The model was trained on the **raw** text split by spaCy's English tokenizer (`common.py`, `train.py`). The app does
exactly that and only trims outer whitespace (as the project's original `gui.py` did). It deliberately does **not**
lowercase, stem or remove stop-words, because that would change the model's input compared with training.

## Project structure
```
app.py                    entry point: theme, top navigation
views/extraction_page.py  Page 1
views/evaluation_page.py  Page 2
pipeline.py               model loading + validation, preprocessing, extraction, gold comparison
results.py                loads real outputs, stand-in detection, runs the evaluation scripts
ui.py                     light-mode CSS, cards, entity highlighting, Plotly styling
common.py train.py evaluate.py explain.py     existing project code (reused)
predict_spacy.py audit.py                     existing audit code (reused)
data/                     train.json val.json test.json labels.txt (from your uploads)
.streamlit/config.toml    pins the light theme
tests/smoke_test.py       renders both pages headlessly and fails on any exception
```
The only change to existing code: in `common.py`, `MODEL_DIR` can be overridden with the `NER_MODEL_DIR`
environment variable (default is still `model`).

## Known limitations (also reported on Page 2 from the audit)
- Only the four attributes above are extracted; brand, colour, size and material are not supported.
- spaCy's NER does not give per-entity confidence scores, so none are shown.
- Gold labels in the dataset are incomplete in places, so some "false alarms" may be genuine mentions (see the label-quality check).
- Several product categories have very few test records; their scores are flagged as unreliable.

## Troubleshooting
- **"spaCy could not load model"** - the pipeline was saved with a different spaCy version. Install the version in `model/meta.json` (`spacy_version`) or retrain with `python train.py`.
- **Page 2 says no results** - run the commands above or use the button on that page.
- **Port in use** - `streamlit run app.py --server.port 8502`.
- **Smoke test** - `python tests/smoke_test.py` (works with or without a model; without one it checks the empty states).
