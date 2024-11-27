install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	make install

build:
	pip install -e ".[release]"
	python -m build --sdist --wheel

test-all:
	make coverage-report
	make type
	make lint

unit-test:
	pytest --mpl -k "not slow"

integration-test:
	pytest --mpl -k "slow"

coverage-report:
	coverage run --branch -m pytest --mpl -k "not slow"
	coverage report

coverage-html:
	coverage run --branch -m pytest --mpl -k "not slow"
	coverage html

lint:
	flake8 bcipy

lint-fix:
	autopep8 --in-place --aggressive --max-line-length 120 --ignore "E402,E226,E24,W50,W690" -r bcipy
	flake8 bcipy

type:
	mypy bcipy

type-html:
	mypy bcipy --html-report type-report 

clean:
	find . -name "*.py[co]" -o -name __pycache__ -exec rm -rf {} +
	find . -path "*/*.pyo"  -delete
	find . -path "*/*.pyc"  -delete
	find . -path "*/*/__pycache__"  -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf bcipy.egg-info
	rm -rf bcipy_cache

bci-gui:
	python bcipy/gui/BCInterface.py

viewer:
	python bcipy/gui/viewer/data_viewer.py --file $(filepath)

offset:
	python bcipy/helpers/offset.py -p

offset-recommend:
	python bcipy/helpers/offset.py -r -p
