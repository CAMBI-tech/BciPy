# Data Acquisition Module (daq)


## Prerequisites

You will need the following things properly installed on your computer.

* [Git](http://git-scm.com/)
* [Python](https://www.python.org)
* [Pip](https://pip.pypa.io/en/stable/)

## Installation

* `git clone <repository-url>` this repository
* change into the new directory

* Install `pip` if it is not already installed on your machine: `sudo easy_install pip`
* Install virtualenv globally: `sudo -H pip install virtualenv`
* Create a virtualenv for your project: `virtualenv venv`
* Activate the environment: `. venv/bin/activate`
* Install dependencies: `pip install -r requirements.txt`

When you are done working in the virtual environment, you can execute the `deactivate` command, which will put you back to the default Python interpreter.

Any time you are working on the project, if the environment is not activated (you can tell by the cursor prompt change), you must `activate` it before running any of the code.

## Running Tests/Code Quality tools

Note: install commands only need to happen once.

* `pip install pytest flake8 isort`
* `pip install -r requirements.txt`
* `flake8 daq`
* `pytest`

By default all doctests as well as module tests are run. You can edit pytest.ini to disable doctests from running.

Alternatively, to only run doctests in a given module:

* `python -m doctest -v daq/core.py`

To sort the import statements according to recommendations, use the `isort` tool.

* `isort daq/client.py` ; sorts single file.
* `isort daq/client.py --diff` ; view sort changes without saving.
* `isort -rc daq` ; sort all files in a directory.

## Style Guidelines

This project conforms to the python [PEP 8](https://www.python.org/dev/peps/pep-0008) style guidelines (with some minor exceptions for long lines). There are some tools to assist with this, including:

* [flake8](https://pypi.python.org/pypi/flake8)
* [autopep8](https://github.com/hhatto/autopep8)
* [isort](https://pypi.python.org/pypi/isort/)