.PHONY: setup run test shots
setup: ; pip install -r requirements.txt
run:   ; python run_pipeline.py --open
test:  ; pytest -q
shots: ; python scripts/make_screenshots.py
