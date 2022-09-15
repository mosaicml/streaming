# several pytest settings
PYTHON ?= python3  # Python command
PYTEST ?= pytest  # Pytest command
PYRIGHT ?= pyright  # Pyright command. Pyright must be installed seperately -- e.g. `node install -g pyright`
EXTRA_ARGS ?=  # extra arguments for pytest

dirs := streaming tests

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
	find streaming tests -type f -name "*.py" | xargs grep -x '.\{100,\}' 1>&2

test:
	$(PYTHON) -m $(PYTEST) $(EXTRA_ARGS)

.PHONY: test lint style
