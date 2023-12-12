# several pytest settings
PYTHON ?= python  # Python command
PYTEST ?= pytest  # Pytest command
PYRIGHT ?= pyright  # Pyright command. Pyright must be installed seperately -- e.g. `node install -g pyright`
EXTRA_ARGS ?=  # extra arguments for pytest

dirs := streaming tests docs

# this only checks for style & pyright, makes no code changes
lint:
	$(PYTHON) -m isort -l 99 -c --diff $(dirs)
	$(PYTHON) -m yapf -dr --style pyproject.toml $(dirs)
	$(PYTHON) -m docformatter -r $(dirs)
	$(PYRIGHT) $(dirs)

# run this to autoformat your code
style:
	$(PYTHON) -m isort -l 99 $(dirs)
	$(PYTHON) -m yapf -rip --style pyproject.toml $(dirs)
	$(PYTHON) -m docformatter -ri $(dirs)

longlines:
	python3 scripts/long_lines.py

test:
	$(PYTHON) -m $(PYTEST) $(EXTRA_ARGS)

web:
	uvicorn scripts.partition.web:app --port 1337 --reload

simulator:
	streamlit run simulation/interfaces/sim_ui.py

.PHONY: test lint style
