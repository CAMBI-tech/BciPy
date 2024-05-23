install:
	pip install -e .

dev-install:
	pip install -r dev_requirements.txt
	pip install psychopy==2023.2.1 --no-deps
	pip install kenlm==0.1 --global-option="--max_order=12"
	make install

make dev-install-mac:
	sh scripts/shell/m2chip_install.sh
	brew install labstreaminglayer/tap/lsl
	make dev-install

test-all:
	make coverage-report
	make type
	make lint

unit-test:
	pytest --mpl -k "not slow"

integration-test:
	pytest --mpl -k "slow"

coverage-report:
	coverage run --branch --source=bcipy -m pytest --mpl -k "not slow"
	coverage report

coverage-html:
	coverage run --branch --source=bcipy -m pytest --mpl -k "not slow"
	coverage html

lint:
	flake8 bcipy

lint-fix:
	autopep8 --in-place --aggressive -r bcipy
	flake8 bcipy

type:
	mypy bcipy

type-html:
	mypy bcipy --html-report type-report 

clean:
	find . -name "*.py[co]" -o -name __pycache__ -exec rm -rf {} +
	find . -path "*/*.pyo"  -delete
	find . -path "*/*.pyc"  -delete

bci-gui:
	python bcipy/gui/BCInterface.py

viewer:
	python bcipy/gui/viewer/data_viewer.py --file $(filepath)
