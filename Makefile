install:
	pip install -e .

dev-install:
	pip install -r dev_requirements.txt
	pip install -e .

test-all:
	coverage run --branch --source=bcipy -m pytest --mpl -k "not slow"
	coverage report
	flake8 bcipy

unit-test:
	pytest -k "not slow"

integration-test:
	pytest -k "slow"

coverage-html:
	coverage run --branch --source=bcipy -m pytest --mpl -k "not slow"
	coverage html

lint:
	autopep8 --in-place --aggressive -r bcipy
	flake8 bcipy

clean:
	find . -name "*.py[co]" -o -name __pycache__ -exec rm -rf {} +
	find . -path "*/*.pyo"  -delete
	find . -path "*/*.pyc"  -delete

bci-gui:
	python bcipy/gui/BCInterface.py

run-with-defaults:
	bcipy