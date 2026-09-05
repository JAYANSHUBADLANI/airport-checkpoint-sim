PY := python3
export PYTHONPATH := src

.PHONY: demo fetch tsa run quick report test clean

demo: fetch tsa test run report
	@echo
	@echo "Tables in outputs/tables, figures in outputs/figures,"
	@echo "formatted blocks in outputs/report_fragments.md."

fetch:
	$(PY) scripts/fetch_data.py

tsa:
	$(PY) scripts/parse_tsa.py

run:
	$(PY) -u scripts/run_all.py

quick:
	$(PY) -u scripts/run_all.py --quick

report:
	$(PY) scripts/make_report.py

test:
	$(PY) -m pytest tests -q

clean:
	rm -rf outputs/tables outputs/figures outputs/report_fragments.md \
	       data/interim .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
	mkdir -p outputs/tables outputs/figures
