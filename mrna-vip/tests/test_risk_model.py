import joblib
import numpy as np
import pytest

from src import config
from src.risk_model import load_model, predict_risk, train_and_select


@pytest.fixture(scope="module")
def trained(tmp_path_factory, clean_df):
    out = tmp_path_factory.mktemp("out")
    return out, train_and_select(clean_df, output_dir=out)


def test_both_models_trained_and_artifacts_saved(trained):  # FR-1.5, FR-1.6
    out, summary = trained
    assert set(summary["models"]) == {"logistic_regression", "random_forest"}
    for name in summary["models"]:
        assert (out / "figures" / f"confusion_matrix_{name}.png").exists()
    assert (out / "classification_reports.txt").read_text().count("precision") == 2


def test_best_model_selected_by_f1_and_reloadable(trained, clean_df):  # FR-1.7
    out, summary = trained
    f1s = {n: m["f1_macro"] for n, m in summary["models"].items()}
    assert summary["selected_model"] == max(f1s, key=f1s.get)
    model = joblib.load(out / "risk_model.joblib")
    X = clean_df[config.FEATURE_COLS].head(20)
    assert np.array_equal(model.predict(X), joblib.load(out / "risk_model.joblib").predict(X))


def test_predict_risk(trained, patient):
    model = joblib.load(trained[0] / "risk_model.joblib")
    label, p = predict_risk(model, patient)
    assert label in {"High", "Low"} and 0 <= p <= 1
    with pytest.raises(KeyError):
        predict_risk(model, {"Age": 3})
    with pytest.raises(ValueError):
        predict_risk(model, {**patient, "Prior_Reaction_History": "Severe"})
