install:
	pip install -e .

dev-install:
	pip install -r dev_requirements.txt
	pip install -e .

test-all:
	coverage run --branch --source=bcipy -m pytest
	flake8 bcipy
	coverage report

coverage-html:
	coverage run --branch --source=bcipy -m pytest
	coverage html

lint:
	autopep8 --in-place --aggressive -r bcipy
	flake8 bcipy

clean:
	find . -name "*.py[co]" -o -name __pycache__ -exec rm -rf {} +
	find . -path "*/*.pyo"  -delete
	find . -path "*/*.pyc"  -delete
