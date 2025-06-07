.PHONY: scrape-now parse index ui fmt test

venv/bin/activate:
	python -m venv venv && . venv/bin/activate && pip install -r requirements.txt

scrape-now:
	python3 -m src.batch_scraper.fetch_all

parse:
	python3 -m src.preprocessing.build_json

index:
	python3 -m src.indexing.build_index

ui:
	streamlit run src/ui/app.py

fmt:
	ruff check --fix .
	black .

test:
	pytest -q
