"""Build one self-contained results page (outputs/index.html): double-click to view, no server needed."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

from src import config

CSS = """body{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1b2430}
h1{margin-bottom:.2rem}h2{border-bottom:2px solid #0F6E6E;padding-bottom:.2rem;margin-top:2rem}
.note{background:#fff4e5;border-left:4px solid #e08a00;padding:.6rem 1rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.8rem}
.card{background:#f3f7f7;border-radius:8px;padding:.8rem;text-align:center}.card b{font-size:1.6rem;display:block;color:#0F6E6E}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd;padding:.3rem .5rem;text-align:left;font-size:13px}
img{max-width:100%;border:1px solid #ddd;border-radius:6px}.no{color:#b00020}.yes{color:#0a7d33}
blockquote{margin:.3rem 0;padding-left:.8rem;border-left:3px solid #0F6E6E;color:#445}"""


def _img(path: Path) -> str:
    return f'<img alt="{path.stem}" src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}">'


def build(output_dir: Path = config.OUTPUT_DIR) -> Path:
    out = Path(output_dir)
    m = json.loads((out / "model_metrics.json").read_text())
    best = m["models"][m["selected_model"]]
    lat = json.loads((out / "latency_log.json").read_text())
    latest = {p.stem.split("_")[0]: p for p in sorted((out / "reports").glob("*.json"))}  # newest per patient
    reports = [json.loads(p.read_text()) | {"_id": pid} for pid, p in latest.items()]
    reports.sort(key=lambda r: not r["Eligible_For_Vaccine"])  # show an eligible case first
    cards = [("Selected model", m["selected_model"].replace("_", " ")), ("Test macro F1", f"{best['f1_macro']:.2f}"),
             ("ROC AUC", f"{best['roc_auc']:.2f}"), ("Patients reported", str(len(reports))),
             ("Retrieval", lat["retrieval_backend"])]
    rows = "".join(f"<tr><td>{n.replace('_', ' ')}</td><td>{r['f1_macro']:.3f}</td><td>{r['roc_auc']:.3f}</td>"
                   f"<td>{r['accuracy']:.2f}</td><td>{r['cv_f1_macro_mean']:.3f}</td></tr>" for n, r in m["models"].items())
    rep = "".join(
        f"<h3>Patient {r['_id']} - <span class='{'yes' if r['Eligible_For_Vaccine'] else 'no'}'>"
        f"{'ELIGIBLE' if r['Eligible_For_Vaccine'] else 'NOT ELIGIBLE'}</span> (dose {r['Recommended_Dose_mg']:g} mg)</h3>"
        "<ol>" + "".join(f"<li>{html.escape(s)}</li>" for s in r["Reasoning_Steps"]) + "</ol>"
        + "".join(f"<blockquote>{html.escape(c)}</blockquote>" for c in r["FDA_Safety_Warnings_Cited"])
        for r in reports[:3])
    retrieval = (out / "retrieval_report.md").read_text().split("\n\n", 2)[-1].strip().splitlines()
    rtable = "<table>" + "".join(
        "<tr>" + "".join(f"<{'th' if i == 0 else 'td'}>{html.escape(c.strip())}</{'th' if i == 0 else 'td'}>"
                         for c in line.strip("|").split("|")) + "</tr>"
        for i, line in enumerate(l for l in retrieval if not set(l) <= set("|- "))) + "</table>"
    fig = out / "figures"
    page = f"""<!doctype html><meta charset=utf-8><title>mRNA-VIP results</title><style>{CSS}</style>
<h1>mRNA Vaccine Clinical Safety &amp; Dosage Intelligence Platform</h1>
<p class=note><b>Student engineering project.</b> {html.escape(config.DISCLAIMER)}</p>
<div class=grid>{''.join(f'<div class=card><b>{html.escape(v)}</b>{k}</div>' for k, v in cards)}</div>
<h2>Dashboard</h2>{_img(fig / 'dashboard.png')}
<h2>Module 1 - classifier comparison (held-out test set)</h2>
<table><tr><th>Model</th><th>Macro F1</th><th>ROC AUC</th><th>Accuracy</th><th>CV F1 (train)</th></tr>{rows}</table>
<p>{_img(fig / 'confusion_matrix_logistic_regression.png')[:-1]} width=320> {_img(fig / 'confusion_matrix_random_forest.png')[:-1]} width=320></p>
<h2>Module 3 - retrieval test queries</h2>{rtable}
<section id=reports><h2>Module 4 - sample safety reports</h2>{rep}</section>"""
    path = out / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


if __name__ == "__main__":
    print(build())
