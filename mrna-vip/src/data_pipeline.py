"""Module 1a - data loading, validation, cleaning and preprocessing.

Satisfies FR-1.1 .. FR-1.4.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src import config


class MissingColumnsError(ValueError):
    """Raised when the input CSV lacks one or more required columns (FR-1.1)."""

    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(f"patient_records is missing required column(s): {', '.join(missing)}")


def load_raw(path: str | Path = config.RAW_CSV) -> pd.DataFrame:
    """Load the CSV and validate the schema.

    NOTE: pandas treats the literal string "None" as NaN by default. In this
    dataset "None" is a *real category* of Prior_Reaction_History (no previous
    reaction), so default NA parsing would silently turn ~60% of that column
    into "missing". Only truly empty cells are treated as missing here.
    """
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise MissingColumnsError(missing)
    return df


def profile(df: pd.DataFrame) -> dict:
    """Shape, dtypes, missing counts and summary statistics (PRD task 1)."""
    return {
        "shape": list(df.shape),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(n) for c, n in df.isnull().sum().items()},
        "summary": df.describe(include="all").round(3).to_dict(),
    }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values (FR-1.2) and sanity-check ranges.

    * Biomarker_Level -> column median. Median (not mean) because the biomarker
      is right-skewed, so the median is robust to the high-value tail.
    * Prior_Reaction_History -> column mode (categorical; ~3.5% missing).
    Imputation (rather than row removal) keeps all 500 rows - the dataset is small.
    Medians/modes are computed on the full file, which is the PRD's cleaning step;
    the *scaler* (below) is fitted on training rows only to avoid leakage.
    """
    out = df.copy()
    out["Biomarker_Level"] = out["Biomarker_Level"].fillna(out["Biomarker_Level"].median())
    mode = out[config.CATEGORICAL_COL].mode(dropna=True)
    out[config.CATEGORICAL_COL] = out[config.CATEGORICAL_COL].fillna(mode.iloc[0])

    unknown = set(out[config.CATEGORICAL_COL]) - set(config.CATEGORY_ORDER)
    if unknown:
        raise ValueError(f"Unexpected Prior_Reaction_History values: {sorted(unknown)}")
    if out[config.ID_COL].duplicated().any():
        raise ValueError("Patient_ID values must be unique")
    return out


def load_clean(path: str | Path = config.CLEAN_CSV) -> pd.DataFrame:
    """Re-read the cleaned CSV. Must keep the literal "None" category (see load_raw)."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def load_and_clean(path: str | Path = config.RAW_CSV) -> pd.DataFrame:
    return clean(load_raw(path))


def build_preprocessor() -> ColumnTransformer:
    """Standardize numerics (FR-1.3) and ordinal-encode the reaction history.

    Ordinal order None < Mild < Moderate reflects severity, which a linear model
    can use directly. Pre_Existing_Conditions is already a small integer count.
    Wrapping this in a sklearn Pipeline means the scaler is fitted on the
    training split only, and the very same transform is reapplied at inference.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_COLS),
            ("cat", OrdinalEncoder(categories=[config.CATEGORY_ORDER]), [config.CATEGORICAL_COL]),
            ("count", "passthrough", [config.COUNT_COL]),
        ]
    )


def split_data(df: pd.DataFrame, seed: int = config.RANDOM_SEED):
    """80/20 stratified split with a fixed seed (FR-1.4). Returns X_train, X_test, y_train, y_test."""
    return train_test_split(
        df[config.FEATURE_COLS],
        df[config.TARGET_COL],
        test_size=0.2,
        random_state=seed,
        stratify=df[config.TARGET_COL],
    )


def main() -> None:
    df = load_and_clean()
    config.CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_CSV, index=False)
    print(f"Saved {config.CLEAN_CSV} - missing values: {int(df.isnull().sum().sum())}")


if __name__ == "__main__":
    main()
