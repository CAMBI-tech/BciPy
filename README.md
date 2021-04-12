# Brain- Computer Interface Codebase
------------------------------------

### What is it?

It is Brain-computer interface software written in Python. It can function as a standalone or you can take the tools you need and start coding your own system. See our official BciPy documentation including affiliations and more context information here: https://bcipy.github.io/  (in progress).

It should, based on our dependencies, work on most recent operating systems, however it has only been verified on Windows (7 & 10 Pro) and Mac OSx (High Sierra & Mojave) at this time. It won't build as is on Linux. Some additional work will be needed to install WxPython and pylsl.

### Contributions Welcome!

This is our first release. It is verified using LSL with DSI and gtec for the Calibration modes only at this time with both image and text stimuli. It comes with a fake data server to help you develop while mocking real time EEG acquisition. We are taking all suggestions at this time for additions, eventually we'll make a contributions wishlist. If you want to be added to the development team, reach out to us and we'll add you to the team slack.

*Please cite us when using!*

Use this citation for now:

```
Memmott, T., Kocanaogullari, A., Erdogmus, D., Bedrick, S., Peters, B., Fried-Oken, M. & Oken, B. (2018, May). BciPy: A Python Framework for Brain-Computer Interface Research. Poster presented at the 7th International BCI meeting 2018 in Asilomar, CA.
```

## Features
-----------

*RSVPKeyboard*

```
	*RSVP KeyboardTM* is an EEG (electroencephalography) based BCI (brain
		computer interface) typing system. It utilizes a visual presentation technique
		called rapid serial visual presentation (RSVP). In RSVP, the options are
		presented rapidly at a single location with a temporal separation. Similarly
		in RSVP KeyboardTM, the symbols (the letters and additional symbols) are
		shown at the center of screen. When the subject wants to select a symbol,
		they await the intended symbol during the presentation and elicit a p300 response to a target symbol.

	To run on windows, run `python bcipy/gui/BCInterface.py` in your terminal to begin.
```

## Dependencies
---------------
This project requires Psychopy, Python v 3.6.5, and other packages. See requirements.txt. When possible integration with other open source libraries will be done.


## Installation
---------------

#### BCI Setup

In order to run BCI suite on your computer, first install **Python 3.6.5** [from here.](https://www.python.org/downloads/)

You must install Docker and Docker-Machine to use the Language Model developed by CSLU. There are instructions in the language model directory for getting the image you need (think of it as a callable server). You'll also need to download and load the language model [images](https://drive.google.com/drive/folders/1OYpUYASAceb60b2c5obyYytEZ0AZrajY?usp=sharing). If not using or rolling your own, set fake_lm to true in the parameters.json file.

To use all the goodies locally (including the GUI and demo scripts)
1. Git clone https://github.com/BciPy/BciPy.git
2. Change directory in your terminal to the repo
3. Run `pip install -e .`
4. If using Mac, you will need to install XCode and enable command line tools. `xcode-select --install`
5. If you're on Windows, you may need to uninstall pygame (`pip uninstall pygame`). Psychopy, for historical reasons, keeps pygame but it just spams your console logs if you only want to use pyglet (which we use in this repository!)

If wanting the latest version from PyPi:
1. `pip install bcipy`

Alternately, if [Make](http://www.mingw.org/) is installed, you may run the follow command to install:

```sh
# install in development mode
make dev-install
```

## Usage Locally

Start by running `python bcipy/gui/BCInterface.py` in your command prompt or terminal. You may also invoke the experiment directly using command line tools for bci_main.py.

Ex. `python bci_main.py` *this will use default parameters, user, experiment and task*

You can pass it attributes with flags, if desired.

Ex. `python bci_main.py --user "bci_user" --task "RSVP Calibration"`

Use the help flag to see other available input options: `python bci_main.py --help`

## Example usage as a package

```python
from bcipy.helpers import system_utils
system_utils.get_system_info()
```

## Modules and Vital Functions
------------------------------

This a list of the major modules and their functionality. Each module will contain its own README, demo and test scripts. Please check them out for more information!

- `acquisition`: acquires data, gives back desired time series, saves to file at end of session.
- `display`: handles display of stimuli on screen and passes back stimuli timing.
- `signal`: eeg signal models, filters, processing, evaluators and viewers. 
- `gui`: end-user interface into registered bci tasks and parameter editing. See BCInterface.py.
- `helpers`: helpful functions needed for interactions between modules, basic I/O, and data visualization. 
- `language_model`: gives probabilities of next letters during typing.
- `parameters`: location of json parameters.
- `static`: image and sound stimuli, misc manuals, and readable texts for gui.
- `tasks`: bcipy implemented user tasks. Main collection of bci modules for use during various experimentation. Ex. RSVPCalibration.
- `feedback`: feedback mechanisms for sound and visual stimuli.
- `bci_main`: executor of experiments. Main entry point into the application

## Demo and Tests
-----------------

All major functions and modules have demo and test files associated with them which may be run locally. This should help orient you to the functionality as well as serve as documentation. *If you add to the repo, you should be adding tests and fixing any test that fail when you change the code.*

For example, you may run the bci_main demo by:

`python demo/bci_main_demo.py`

This demo will load in parameters and execute a demo task defined in the file. There are demo files for all modules listed above except language_model, helpers, and utils. Run them as a python script!

This repository uses pytest for execution of tests. You may execute them by:

`py.test` or `pytest` depending on your OS

## Contribution Guidelines
--------------------------

We follow and will enforce the contributor's covenant to foster a safe and inclusive environment for this open source software, please reference this link for more information: https://www.contributor-covenant.org/

1. All added code will need tests and a demo (if a large feature).
2. All tests must pass to merge, even if they are seemingly unrelated to your task.
3. Pull requests must be tested locally and by the requester on a different computer.
4. Use Spaces, not Tabs.
5. Use informative names for functions and classes.
6. Document the input and output of your functions / classes in the code. eg in-line commenting
7. Do not push IDE or other local configuration files.
8. All new modules or major functionality should be documented outside of the code with a README.md. See README.md in repo or go to this site for inspiration: https://github.com/matiassingers/awesome-readme. Always use a Markdown interpreter before pushing. There are many free online or your IDE may come with one.

Use this resource for examples: http://docs.python-guide.org/en/latest/writing/style/

## Testing
----------

When writing tests, put them in the correct module, in a tests folder, and prefix the file and test itself with `test` in order for pytest to discover it. See other module tests for examples!

Development requirements must be installed before running: `pip install dev_requirements.txt`

To run all tests, in the command line:

```python
py.test
```


To run a single modules tests (ex. acquisition), in the command line:

```python
py.test acquisition
```

To generate test coverage metrics, in the command line:

```bash
coverage run --branch --source=bcipy -m pytest

#Generate a command line report
coverage report

# Generate html doc in the bci folder. Navigate to index.html and click.
coverage html
```

Alternately, if Make is installed, you may run the follow command to run coverage/pytest and generate the html:

```sh
make coverage-html
```

## Linting

This project enforces `PEP` style guidelines using [flake8](http://flake8.pycqa.org/en/latest/).

To avoid spending unnecessary time on formatting, we recommend using `autopep8`. You can specify a file or directory to auto format. When ready to push your code, you may run the following commands to format your code:

```sh
# autoformat all files in bcipy
autopep8 --in-place --aggressive -r bcipy

# autoformat only the processor file
autopep8 --in-place --aggressive bcipy/acquisition/processor.py
```

Finally, run the lint check: `flake8 bcipy`.

Alternately, if Make is installed, you may run the follow command to run autopep8 and flake8:

```sh
make lint
```

## Glossary
-----------

***Stimuli***: A single letter, tone or image shown (generally in an inquiry). Singular = stimulus, plural = stimuli.

***Trial***: A collection of data after a stimuli is shown. A----

***Inquiry***: The set of stimuli after a fixation cross in a spelling task to gather user intent. A ---- B --- C ----

***Series***: Each series contains at least one inquiry. A letter/icon decision is made after a series in a spelling task.

***Session***: Data collected for a task. Comprised of metadata about the task and a list of Series.


## Authorship
--------------

- Tab Memmott (OHSU)
- Matthew Lawhead (OHSU)
- Aziz Kocanaogullari (NEU)
- Shiran Dudy (OHSU)
- Dani Smektala (OHSU)
- Ian Jackson (Reed)
- Alister Cedeño (OHSU)
- Berkan Kadioglu (NEU)
- Basak Celik (NEU)
- Andac Demir (NEU)
- Shaobin Xu (NEU)
