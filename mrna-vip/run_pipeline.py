"""Run the whole platform: python run_pipeline.py [--provider offline|openai|anthropic] [--open]"""
import argparse
import sys
import time
import webbrowser

from src import attention, config, dashboard, rag_index, report_generator, risk_model, showcase


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None, help="LLM provider (default: LLM_PROVIDER env or offline)")
    ap.add_argument("--open", action="store_true", help="open the results page in your browser")
    args = ap.parse_args()
    for d in (config.FIGURE_DIR, config.REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0]] + (["--provider", args.provider] if args.provider else [])

    stages = [("1 Data + risk classifier", risk_model.main), ("2 Self-attention", attention.run),
              ("3 RAG index", rag_index.main), ("4 Structured reports", report_generator.main),
              ("5 Dashboard", dashboard.build_dashboard), ("6 Results page", showcase.build)]
    timings = []
    for name, fn in stages:
        print(f"\n=== Module {name} ===")
        t = time.perf_counter()
        out = fn()
        if hasattr(out, 'suffix'):
            print(f"Saved {out}")
        timings.append((name, time.perf_counter() - t))
    print("\n--- Timings ---")
    for name, s in timings:
        print(f"{name:<28}{s:6.2f}s")
    print(f"{'Total':<28}{sum(s for _, s in timings):6.2f}s")
    page = config.OUTPUT_DIR / "index.html"
    print(f"\nOpen the results page: {page}")
    if args.open:
        webbrowser.open(page.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
