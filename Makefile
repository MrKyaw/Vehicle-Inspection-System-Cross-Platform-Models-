PYTHON ?= python3
COV_ARGS = --cov=vehicle_inspection --cov-branch --cov-report=term-missing

.PHONY: help install test coverage demo json smoke clean

help:
	@echo "Vehicle Inspection System"
	@echo "  make install   Install the package with dev dependencies"
	@echo "  make test      Run the test suite"
	@echo "  make coverage  Run tests with the 100% branch coverage gate"
	@echo "  make demo      Run the built-in inspection scenario"
	@echo "  make json      Run the demo and print the JSON report"
	@echo "  make smoke     Check CLI help and demo entry points"
	@echo "  make clean     Remove test and coverage artifacts"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

coverage:
	$(PYTHON) -m pytest $(COV_ARGS)

demo:
	$(PYTHON) -m vehicle_inspection --demo

json:
	$(PYTHON) -m vehicle_inspection --demo --json

smoke:
	$(PYTHON) -m vehicle_inspection --help > /dev/null
	$(PYTHON) -m vehicle_inspection --demo > /dev/null
	@echo "CLI smoke checks passed"

clean:
	rm -rf .pytest_cache .coverage .coverage.* htmlcov coverage.xml build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
