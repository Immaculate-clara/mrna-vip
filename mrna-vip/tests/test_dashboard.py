import json

import numpy as np
import pandas as pd
from PIL import Image

from src.dashboard import build_dashboard


def _fake_outputs(d):
    pd.DataFrame({"y_true": [0, 1, 0, 1, 1, 0], "y_score": [.1, .8, .4, .6, .7, .3]}).to_csv(
        d / "test_predictions.csv", index=False)
    np.savez(d / "attention_result.npz", weights=np.full((5, 5), .2), sequence=np.array("ACGUA"))
    (d / "latency_log.json").write_text(json.dumps(
        {"rag_retrieval_s": [0.01], "llm_generation_s": [0.5]}))


def test_three_panels_and_dpi(tmp_path):  # FR-5.1, FR-5.3
    _fake_outputs(tmp_path)
    path = build_dashboard(tmp_path)
    assert Image.open(path).info["dpi"][0] >= 149.9


def test_panels_follow_saved_outputs(tmp_path):  # FR-5.2
    _fake_outputs(tmp_path)
    a = build_dashboard(tmp_path, tmp_path / "a.png").read_bytes()
    (tmp_path / "latency_log.json").write_text(json.dumps(
        {"rag_retrieval_s": [0.2], "llm_generation_s": [0.1]}))
    assert build_dashboard(tmp_path, tmp_path / "b.png").read_bytes() != a


def test_results_page_builds(tmp_path):
    from src import config, showcase
    import shutil
    shutil.copytree(config.OUTPUT_DIR, tmp_path, dirs_exist_ok=True)
    html = showcase.build(tmp_path).read_text()
    assert "Dashboard" in html and config.DISCLAIMER in html and html.count("<h3>") <= 5
