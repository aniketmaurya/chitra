SRC = $(wildcard ./*.ipynb)

all: chitra docs

docs_serve:
	mkdocs serve

test:
	pytest

clean:
	rm -rf dist

style:
	yapf -ir chitra
	isort chitra

dist: clean
	flit build

pypi: dist
	flit publish

push:
	git push && git push --tags