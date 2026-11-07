"""Module 1b - train, evaluate, select and serialize the risk classifier.

Satisfies FR-1.5 .. FR-1.7.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src import config
from src.data_pipeline import build_preprocessor, load_and_clean, split_data

HIGH_RISK_THRESHOLD = 0.5


def make_models(seed: int = config.RANDOM_SEED) -> dict[str, Pipeline]:
    """Both models share identical preprocessing and training data (FR-1.5)."""
    return {
        "logistic_regression": Pipeline(
            [("prep", build_preprocessor()), ("clf", LogisticRegression(max_iter=1000))]
        ),
        "random_forest": Pipeline(
            [("prep", build_preprocessor()),
             ("clf", RandomForestClassifier(n_estimators=200, random_state=seed))]
        ),
    }


def evaluate(model: Pipeline, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]
    return {
        "f1_macro": float(f1_score(y_test, preds, average="macro")),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "accuracy": float((preds == np.asarray(y_test)).mean()),
        "classification_report": classification_report(y_test, preds, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }


def save_confusion_matrix(model: Pipeline, X_test, y_test, name: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["Low risk", "High risk"], cmap="Blues", ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Confusion matrix - {name}")
    fig.tight_layout()
    path = out_dir / f"confusion_matrix_{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def train_and_select(df: pd.DataFrame, output_dir: Path = config.OUTPUT_DIR,
                     seed: int = config.RANDOM_SEED) -> dict:
    """Train both models, write all artifacts, serialize the F1 winner (FR-1.7)."""
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = split_data(df, seed)
    models = make_models(seed)
    results: dict[str, dict] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
        results[name]["cv_f1_macro_mean"] = float(
            cross_val_score(make_models(seed)[name], X_train, y_train, cv=5, scoring="f1_macro").mean()
        )
        save_confusion_matrix(model, X_test, y_test, name, fig_dir)
        print(f"--- {name} ---")
        print(classification_report(y_test, model.predict(X_test),
                                    target_names=["Low risk", "High risk"]))
        print(np.array(results[name]["confusion_matrix"]))

    best = max(results, key=lambda n: results[n]["f1_macro"])
    best_model = models[best]
    joblib.dump(best_model, output_dir / "risk_model.joblib")

    pd.DataFrame({
        config.ID_COL: df.loc[X_test.index, config.ID_COL].values,
        "y_true": np.asarray(y_test),
        "y_score": best_model.predict_proba(X_test)[:, 1],
        "y_pred": best_model.predict(X_test),
    }).to_csv(output_dir / "test_predictions.csv", index=False)

    summary = {"selected_model": best, "selection_metric": "test-set macro F1",
               "random_seed": seed, "n_train": len(X_train), "n_test": len(X_test),
               "models": results}
    (output_dir / "model_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "classification_reports.txt").write_text(
        "\n\n".join(
            f"=== {n} ===\n" + classification_report(y_test, models[n].predict(X_test),
                                                     target_names=["Low risk", "High risk"])
            + f"\nConfusion matrix:\n{np.array(r['confusion_matrix'])}"
            for n, r in results.items()), encoding="utf-8")
    print(f"Selected model: {best} (macro F1 {results[best]['f1_macro']:.3f})")
    return summary


def load_model(path: Path = config.MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def predict_risk(model: Pipeline, patient: dict) -> tuple[str, float]:
    """Score one patient dict -> ("High"|"Low", probability of high reaction risk)."""
    missing = [c for c in config.FEATURE_COLS if c not in patient]
    if missing:
        raise KeyError(f"Patient is missing feature(s): {missing}")
    if patient[config.CATEGORICAL_COL] not in config.CATEGORY_ORDER:
        raise ValueError(f"{config.CATEGORICAL_COL} must be one of {config.CATEGORY_ORDER}, "
                         f"got {patient[config.CATEGORICAL_COL]!r}")
    prob = float(model.predict_proba(pd.DataFrame([patient])[config.FEATURE_COLS])[0, 1])
    return ("High" if prob >= HIGH_RISK_THRESHOLD else "Low"), prob


def main() -> None:
    df = load_and_clean()
    config.CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_CSV, index=False)
    train_and_select(df)


if __name__ == "__main__":
    main()
