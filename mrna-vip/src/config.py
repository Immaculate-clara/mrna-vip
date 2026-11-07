"""Central configuration: paths, seeds and shared constants.

Every module imports its paths from here so the whole pipeline can be relocated
(or redirected into a temp directory by the tests) in one place.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"

RAW_CSV = DATA_DIR / "patient_records.csv"
CLEAN_CSV = DATA_DIR / "patient_records_clean.csv"
GUIDELINES_TXT = DATA_DIR / "mrna_safety_guidelines.txt"

MODEL_PATH = OUTPUT_DIR / "risk_model.joblib"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
ATTENTION_PATH = OUTPUT_DIR / "attention_result.npz"
VECTOR_STORE_DIR = OUTPUT_DIR / "chroma_store"
RETRIEVAL_LOG_PATH = OUTPUT_DIR / "retrieval_log.jsonl"
LATENCY_LOG_PATH = OUTPUT_DIR / "latency_log.json"

RANDOM_SEED = 42

# ---- Dataset schema (PRD section 5.2) -------------------------------------
REQUIRED_COLUMNS = [
    "Patient_ID",
    "Age",
    "Biomarker_Level",
    "Pre_Existing_Conditions",
    "Vaccine_Dose_mcg",
    "Prior_Reaction_History",
    "Reaction_Score",
]
NUMERIC_COLS = ["Age", "Biomarker_Level", "Vaccine_Dose_mcg"]
COUNT_COL = "Pre_Existing_Conditions"
CATEGORICAL_COL = "Prior_Reaction_History"
CATEGORY_ORDER = ["None", "Mild", "Moderate"]  # ordinal: severity of prior reaction
TARGET_COL = "Reaction_Score"
ID_COL = "Patient_ID"
FEATURE_COLS = NUMERIC_COLS + [CATEGORICAL_COL, COUNT_COL]

# ---- RAG (PRD section 7) --------------------------------------------------
CHUNK_MIN_TOKENS = 300
CHUNK_TARGET_MAX_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "mrna_safety_guidelines"
DEFAULT_TOP_K = 3

# ---- Report generation (PRD section 8) ------------------------------------
DISCLAIMER = (
    "Student engineering project output - not real clinical guidance. "
    "This report must never be used for real medical decision-making."
)
MCG_PER_MG = 1000.0

# Project assumption, NOT a clinical fact: the guideline says "meaningfully above
# the normal reference range" without giving a number. For the offline report
# engine we treat the dataset's 75th percentile (mg/L) as "elevated".
ELEVATED_BIOMARKER_MG_L = 7.0
