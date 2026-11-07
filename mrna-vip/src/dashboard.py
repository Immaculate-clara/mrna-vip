"""Module 5 - three-panel results dashboard built from the other modules' SAVED outputs.

Panel A: ROC curve of the selected classifier   (outputs/test_predictions.csv)
Panel B: self-attention heatmap                  (outputs/attention_result.npz)
Panel C: RAG vs LLM latency                      (outputs/latency_log.json)

Nothing is hardcoded: re-running a module and then this script changes the panels (FR-5.2).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from src import config
from src.attention import plot_attention

DPI = 150  # FR-5.3 requires >= 150


def build_dashboard(output_dir: Path = config.OUTPUT_DIR, save_path: Path | None = None) -> Path:
    output_dir = Path(output_dir)
    save_path = Path(save_path or output_dir / "figures" / "dashboard.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(output_dir / "test_predictions.csv")
    attn = np.load(output_dir / "attention_result.npz")
    latency = json.loads((output_dir / "latency_log.json").read_text(encoding="utf-8"))
    metrics_path = output_dir / "model_metrics.json"
    model_name = json.loads(metrics_path.read_text())["selected_model"] if metrics_path.exists() else "model"

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))

    # Panel A - ROC
    fpr, tpr, _ = roc_curve(preds["y_true"], preds["y_score"])
    axes[0].plot(fpr, tpr, lw=2, color="#0F6E6E", label=f"AUC = {auc(fpr, tpr):.2f}")
    axes[0].plot([0, 1], [0, 1], ls="--", color="gray", label="Chance")
    axes[0].set(title=f"A. Risk Classifier ROC ({model_name.replace('_', ' ')})",
                xlabel="False positive rate", ylabel="True positive rate")
    axes[0].legend(loc="lower right")

    # Panel B - attention
    plot_attention(attn["weights"], str(attn["sequence"]), None, ax=axes[1])
    axes[1].set_title("B. mRNA Self-Attention Weights")

    # Panel C - latency (linear axis starting at zero, so bar heights are honest)
    means_ms = {"RAG retrieval": float(np.mean(latency["rag_retrieval_s"])) * 1000,
                "LLM generation": float(np.mean(latency["llm_generation_s"])) * 1000}
    bars = axes[2].bar(list(means_ms), list(means_ms.values()), color=["#0F6E6E", "#1F3A5F"])
    axes[2].set_ylim(0, max(means_ms.values()) * 1.2)
    axes[2].set(title="C. Pipeline Stage Latency (mean per patient)", ylabel="Milliseconds")
    for bar, v in zip(bars, means_ms.values()):
        axes[2].annotate(f"{v:.1f} ms", (bar.get_x() + bar.get_width() / 2, v),
                         ha="center", va="bottom", fontsize=10)
    axes[2].set_xlabel(f"LLM: {latency.get('llm_client', '?')} | retrieval: {latency.get('retrieval_backend', '?')}",
                       fontsize=8)

    fig.suptitle("mRNA Vaccine Clinical Safety & Dosage Intelligence Platform - student project, "
                 "not clinical guidance", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    print(f"Saved {build_dashboard()}")
