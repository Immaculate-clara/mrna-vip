"""Regenerate docs/screenshots/*.png from a real run (needs playwright + chromium; dev tool only)."""
import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "docs" / "screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)


def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    text = re.sub(r"^.*(UserWarning|warnings\.warn).*\n?", "", r.stdout + r.stderr, flags=re.M)
    return text.strip()


def terminal(title, cmd, text):
    body = html.escape(f"$ {cmd}\n{text}")
    return (f"<body style='margin:0;background:#1e1e1e'><div style='background:#333;color:#ccc;font:13px sans-serif;"
            f"padding:6px 12px'>● ● ●  {title}</div><pre style='margin:0;padding:14px;color:#d4d4d4;"
            f"font:13px/1.4 Consolas,monospace;width:1000px;white-space:pre-wrap'>{body}</pre>")


def main():
    pipeline = run([sys.executable, "run_pipeline.py"])
    tests = run([sys.executable, "-m", "pytest", "-q"])
    fig = ROOT / "outputs" / "figures"
    for src, dst in [("dashboard.png", "04_dashboard.png"), ("attention_heatmap.png", "05_attention_heatmap.png"),
                     ("confusion_matrix_logistic_regression.png", "06_confusion_matrix.png")]:
        shutil.copy(fig / src, SHOTS / dst)
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1040, "height": 700})
        for name, title, cmd, text in [("01_pipeline_run", "Terminal - full pipeline", "python run_pipeline.py", pipeline),
                                       ("02_tests", "Terminal - tests", "pytest", tests)]:
            page.set_content(terminal(title, cmd, text))
            page.screenshot(path=SHOTS / f"{name}.png", full_page=True)
        page.goto((ROOT / "outputs" / "index.html").as_uri())
        page.screenshot(path=SHOTS / "03_results_page.png", full_page=True)
        page.locator("h2:has-text('Module 3') + table").screenshot(path=SHOTS / "07_retrieval.png")
        page.locator("#reports").screenshot(path=SHOTS / "08_sample_report.png")
        b.close()
    print("\n".join(sorted(p.name for p in SHOTS.iterdir())))


if __name__ == "__main__":
    main()
