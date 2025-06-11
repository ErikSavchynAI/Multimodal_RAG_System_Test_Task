# Makefile ─ project tasks
# Usage:  make <target>
# Run `make help` to see a one-liner for every target.

.DEFAULT_GOAL := help                # show help if no target supplied
VENV ?= .venv                        # override with `make VENV=…`
PY   := $(VENV)/bin/python

.PHONY: scrape parse index ui fmt test venv install clean help

# ----------------------------------------------------------------------
# environment & deps
venv:                                ## Create virtual environment (empty)
	python -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip

install: venv                        ## Install Python dependencies
	$(VENV)/bin/pip install -r requirements.txt

# ----------------------------------------------------------------------
# pipeline stages
scrape:                              ## Stage-1: download HTML + images
	$(PY) -m src.batch_scraper.fetch_all

parse: scrape                        ## Stage-2: HTML → JSONL
	$(PY) -m src.preprocessing.build_json

index: parse                         ## Stage-3: build vector indexes
	$(PY) -m src.indexing.build_index

# ----------------------------------------------------------------------
# dev utilities
ui:                                  ## Launch Streamlit demo
	streamlit run ui_app.py

clean:                               ## Remove all generated data
	rm -rf data/raw data/processed data/index

help:                                ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'
