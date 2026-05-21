PYTHON ?= python

.PHONY: demo test clean

demo:
	$(PYTHON) run_analysis.py --config config/research_question.yaml --reports-dir reports

test:
	$(PYTHON) -m unittest discover -s tests

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('reports/tables'), pathlib.Path('reports/figures')]]"
