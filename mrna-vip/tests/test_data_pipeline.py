import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_pipeline import (MissingColumnsError, build_preprocessor, load_raw, split_data)


def test_missing_column_raises_named_error(tmp_path):  # FR-1.1
    df = pd.read_csv(config.RAW_CSV, keep_default_na=False).drop(columns=["Biomarker_Level"])
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(MissingColumnsError, match="Biomarker_Level"):
        load_raw(p)


def test_none_category_is_not_treated_as_missing():
    raw = load_raw()
    assert (raw[config.CATEGORICAL_COL] == "None").sum() > 200
    assert raw[config.CATEGORICAL_COL].isnull().sum() == 17


def test_cleaning_removes_all_missing(clean_df):  # FR-1.2
    assert clean_df.isnull().sum().sum() == 0 and len(clean_df) == 500


def test_scaling_zero_mean_unit_std(clean_df):  # FR-1.3
    X_train, *_ = split_data(clean_df)
    prep = build_preprocessor().fit(X_train)
    scaled = prep.transform(X_train)[:, :3].astype(float)
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(scaled.std(axis=0), 1, atol=1e-9)


def test_split_is_80_20_stratified_and_reproducible(clean_df):  # FR-1.4
    a, b = split_data(clean_df), split_data(clean_df)
    assert list(a[1].index) == list(b[1].index)
    assert len(a[1]) == 100 and len(a[0]) == 400
    assert abs(a[2].mean() - a[3].mean()) < 0.03
