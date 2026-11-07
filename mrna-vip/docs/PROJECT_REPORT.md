# mRNA Vaccine Clinical Safety & Dosage Intelligence Platform - Project Report

> Student engineering project; not clinical guidance.

## 1. Executive summary
This project builds a five-module pipeline that mirrors how regulated-industry AI systems are assembled: a
classical classifier flags a patient's reaction risk, a from-scratch self-attention engine demonstrates the
Transformer core on mRNA sequences, a retrieval index grounds answers in a guideline document, a validated
report generator combines risk and retrieved text into a schema-checked JSON/Markdown safety report, and a
dashboard summarises everything. It demonstrates data cleaning, model evaluation, NumPy maths, retrieval,
structured generation with hallucination guardrails, and testing (39 automated tests).

## 2. Architecture
```
CSV -> Module 1 (clean + classify) -> risk -------------------------+
       Module 2 (self-attention)  -> heatmap ---------------------+|
TXT -> Module 3 (chunk, embed, index) -> top-k chunks -+          ||
                                                       v          vv
                              Module 4 (prompt -> LLM -> validate -> guardrails) -> JSON + MD
                                                       |
                                                       v
                                              Module 5 (3-panel dashboard)
```

## 3. Module 1 - risk classifier
500 rows; 20 missing biomarkers (median-imputed) and 17 missing reaction histories (mode-imputed). An
important pitfall: pandas reads the literal category `None` as missing, which would have blanked 291 valid rows.

| Model (test, n=100) | Macro F1 | ROC AUC | Accuracy | 5-fold CV F1 (train) |
|---|---|---|---|---|
| **Logistic regression (selected)** | **0.719** | **0.767** | 0.72 | 0.744 |
| Random forest (200 trees) | 0.650 | 0.729 | 0.65 | 0.702 |

Confusion matrix (selected, rows = true low/high): `[[39, 12], [16, 33]]`. Logistic regression won because the
signal is noisy and largely additive, which a regularised linear model captures; with 400 training rows the forest
fits the noise. Both the test split and cross-validation agree on the ranking. A 100-row test set is small, so
differences of a few points are within noise; AUC 0.77 is a believable, non-suspicious level.

## 4. Module 2 - self-attention
`softmax(QK^T/sqrt(d_k))V` implemented in NumPy with seeded random embeddings and projections. Every row sums to 1
(tested). The heatmap is almost uniform (weights 0.065-0.069 vs. 1/15 = 0.067) because the random projections are
scaled by 0.1; the only structure is that positions with the same base have identical rows, since attention without
positional encoding cannot tell positions apart. Sinusoidal positional encoding (optional flag) breaks that
symmetry. Nothing is learned, so the plot shows the mechanics, not biological meaning. A 3-token worked example
that can be checked by hand is in `notebooks/02_attention_math.ipynb`.

## 5. Module 3 - retrieval
The guideline is about 1,000 tokens, so the PRD's 300-500 token window yields only **2 chunks** (479 and 497
tokens, ~50-token sentence overlap). Test queries (offline TF-IDF backend; see `outputs/retrieval_report.md`):
storage, 65+ dosing, adverse-event timing, contraindications, monitoring time and dose-review all returned the
chunk containing the answer; latency about 1 ms.

**Failure case:** "Is the vaccine safe during pregnancy?" returns a chunk with a low similarity score (0.13), but
the guideline says nothing about pregnancy. Retrieval always returns *something*; it cannot say "not covered".
Mitigation: a minimum-score threshold or an LLM relevance check, and the generator's rule that insufficient
evidence means "not eligible". With only two large chunks, top-k=3 returns everything, so retrieval adds little
discrimination on this document; smaller chunks would show the retriever working harder.

## 6. Module 4 - structured report
Pipeline: profile queries -> retrieval -> prompt (risk + chunks only) -> LLM -> JSON schema validation + verbatim
citation check (one retry) -> guardrails -> `.json` and `.md`. Example (patient P0141, ML risk High, p = 1.00,
age 84, 4 conditions, dose 144.1 mcg, prior moderate reaction): **not eligible, 0 mg**, with seven verbatim
guideline citations (dose above 100 mcg, 65+ with 2+ conditions, 3+ conditions, prior moderate reaction, elevated
biomarker, dose+biomarker, monitoring). The full file is in `outputs/reports/`. This run used the offline rule
engine; swap in OpenAI/Anthropic with `LLM_PROVIDER`.

## 7. Dashboard
`outputs/figures/dashboard.png` (150 DPI): ROC curve (AUC 0.77), attention heatmap, and mean RAG vs LLM
latency, each drawn from saved module outputs.

## 8. Limitations and future work
1. **Tiny, synthetic data** (500 rows, 100 test): wide confidence intervals. Use repeated CV and calibration; more data.
2. **Retrieval over a ~1,000-token document** and an approximate tokenizer; no "not covered" detection. Add score
   thresholds, a real tokenizer, hybrid search and an evaluation set with labelled answers.
3. **LLM cost/variability and offline stand-in:** the offline engine is deterministic rules, not a language model.
   Run the real provider, log token costs, add an eval harness for faithfulness.
4. **Assumed biomarker threshold** (7.0 mg/L) is not from the guideline; a real system needs a sourced reference range.
5. **The ML risk and the guideline are separate signals** with no clinical validation; never use for real decisions.

## 9. Reproduction
```
pip install -r requirements.txt
python run_pipeline.py
pytest
```
