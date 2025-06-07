.PHONY: scrape-now parse index ui fmt test

# helper: always run the python inside .venv
PY = .venv/bin/python

scrape-now:
	$(PY) -m src.batch_scraper.fetch_all

parse:
	$(PY) -m src.preprocessing.build_json

index:
	$(PY) -m src.indexing.build_index

ui:
	streamlit run src/ui/app.py

fmt:
	ruff check --fix .
	black .

test:
	pytest -q
