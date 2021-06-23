SRC = $(wildcard ./*.ipynb)

all: chitra docs

build_docs:
	cp README.md docs/index.md

docs_serve:
	mkdocs serve

test:
	pytest

coverage:  ## Run tests with coverage
		coverage erase
		coverage run -m pytest
		coverage report -m
		coverage xml

clean:
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

style:
	yapf -ir chitra tests
	isort chitra tests

dist: clean
	flit build

pypi: dist
	flit publish

push:
	git push && git push --tags
