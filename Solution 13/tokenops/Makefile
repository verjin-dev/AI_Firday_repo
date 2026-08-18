.PHONY: install api ui sim benchmark demo test all clean

install:
	pip install -r requirements.txt

api:
	uvicorn backend.main:app --host 127.0.0.1 --port 8013 --reload

ui:
	streamlit run frontend/app.py

sim:
	python scripts/simulate_workload.py

benchmark:
	python scripts/benchmark.py

demo:
	python scripts/demo_flow.py --offline

test:
	python -m pytest tests/ -q

all: sim benchmark test

clean:
	rm -rf data/db/*.db data/db/*.db-journal benchmark_results.json benchmark_chart.png logs
