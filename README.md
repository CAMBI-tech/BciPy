# BciPy: Brain-Computer Interface Software in Python


[![BciPy](https://github.com/CAMBI-tech/BciPy/actions/workflows/main.yml/badge.svg)](https://github.com/CAMBI-tech/BciPy/actions/workflows/main.yml)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/CAMBI-tech/BciPy/fork)
[![Follow on Twitter](https://img.shields.io/twitter/follow/cambi_tech?label=Follow&style=social)](https://twitter.com/cambi_tech)


BciPy is a library for conducting Brain-Computer Interface experiments in Python. It functions as a standalone application for experimental data collection or you can take the tools you need and start coding your own system. See our official BciPy documentation including affiliations and more context information [here](https://bcipy.github.io/) (in progress).

It will run on the latest windows (7, 10, 11), linux (ubuntu 22.04) and macos (Big Sur). Other versions may work as well, but are not guaranteed. To see supported versions and operating systems as of this release see here: [BciPy Builds](https://github.com/CAMBI-tech/BciPy/actions/workflows/main.yml).

*Please cite us when using!*

```
Memmott, T., Koçanaoğulları, A., Lawhead, M., Klee, D., Dudy, S., Fried-Oken, M., & Oken, B. (2021). BciPy: brain–computer interface software in Python. Brain-Computer Interfaces, 1-18.
```

## Dependencies
---------------
This project requires Python 3.7 or 3.8. All other dependencies defined in the requirements.txt.


## Installation
---------------

#### BciPy Setup

In order to run BciPy on your computer, first install **Python 3** [from here.](https://www.python.org/downloads/)

To use all the goodies locally (including the GUI and demo scripts)
1. Git clone https://github.com/BciPy/BciPy.git
2. Change directory in your terminal to the repo
3. Run `pip install -e .`
4. If using Mac, you will need to install XCode and enable command line tools. `xcode-select --install`
5. If you're on Windows, you may need to uninstall pygame (`pip uninstall pygame`). Psychopy, for historical reasons, keeps pygame but it just spams your console logs if you only want to use pyglet (which we use in this repository!)
6. To use the KenLMLanguageModel class, you must manually install the kenlm package. `pip install kenlm==0.1 --global-option="--max_order=12"`.

If wanting the latest version from PyPi:
1. `pip install bcipy`

Alternately, if [Make](http://www.mingw.org/) is installed, you may run the follow command to install:

```sh
# install in development mode
make dev-install
```

#### Usage Locally

Start by running `python bcipy/gui/BCInterface.py` in your command prompt or terminal. This will run the GUI. You may also use the command `make bci-gui`. You may also invoke the experiment directly using command line tools from bcipy.

Ex. `bcipy` *this will use default parameters, user, experiment and task*

You can pass it attributes with flags, if desired.

Ex. `bcipy --user "bci_user" --task "RSVP Calibration"`

Use the help flag to see other available input options: `bcipy --help`

##### Example usage as a package

```python
from bcipy.helpers import system_utils
system_utils.get_system_info()
```

## Glossary
-----------

***Stimuli***: A single letter, tone or image shown (generally in an inquiry). Singular = stimulus, plural = stimuli.

***Trial***: A collection of data after a stimuli is shown. A----

***Inquiry***: The set of stimuli after a fixation cross in a spelling task to gather user intent. A ---- B --- C ----

***Series***: Each series contains at least one inquiry. A letter/icon decision is made after a series in a spelling task.

***Session***: Data collected for a task. Comprised of metadata about the task and a list of Series.

***Task***: An experimental design with stimuli, trials, inquiries and series for use in BCI. For instance, "RSVP Calibration" is a task.

***Mode***: Common design elements between task types. For instance, Calibration and Free Spelling are modes.

***Paradigm***: Display paradigm with unique properties and modes. Ex. Rapid-Serial Visual Presentation (RSVP), Matrix Speller, Steady-State Visual Evoked Potential (SSVEP).


## Core Modules
---------------

This a list of the major modules and their functionality. Each module will contain its own README, demo and tests. Please check them out for more information!

- `acquisition`: acquires data, gives back desired time series, saves to file at end of session.
- `display`: handles display of stimuli on screen and passes back stimuli timing.
- `signal`: eeg signal models, filters, processing, evaluators and viewers.
- `gui`: end-user interface into registered bci tasks and parameter editing. See BCInterface.py.
- `helpers`: helpful functions needed for interactions between modules, basic I/O, and data visualization.
- `language`: gives probabilities of next symbols during typing.
- `parameters`: location of json parameters. This includes parameters.json (main experiment / app configuration) and device.json (device registry and configuration).
- `static`: image and sound stimuli, misc manuals, and readable texts for gui.
- `task`: bcipy implemented user tasks. Main collection of bci modules for use during various experimentation. Ex. RSVP Calibration.
- `feedback`: feedback mechanisms for sound and visual stimuli.
- `main`: executor of experiments. Main entry point into the application
- `config`: configuration parameters for the application, including paths and data filenames.


## Paradigms
------------


> RSVPKeyboard


```
*RSVP KeyboardTM* is an EEG (electroencephalography) based BCI (brain computer interface) typing system. It utilizes a visual presentation technique called rapid serial visual presentation (RSVP). In RSVP, the options are presented rapidly at a single location with a temporal separation. Similarly in RSVP KeyboardTM, the symbols (the letters and additional symbols) are shown at the center of screen. When the subject wants to select a symbol, they await the intended symbol during the presentation and elicit a p300 response to a target symbol.
```

Citation: 
```
Orhan, U., Hild, K. E., 2nd, Erdogmus, D., Roark, B., Oken, B., & Fried-Oken, M. (2012). RSVP Keyboard: An EEG Based Typing Interface. Proceedings of the ... IEEE International Conference on Acoustics, Speech, and Signal Processing. ICASSP (Conference), 10.1109/ICASSP.2012.6287966. https://doi.org/10.1109/ICASSP.2012.6287966
```

> Matrix Speller

```
Matrix Speller is an EEG (electroencephalography) based BCI (brain computer interface) typing system. It utilizes a visual presentation technique called Single Character Presentation (SCP). In matrix speller, the symbols are arranged in a matrix with fixed number of rows and columns. Using SCP, subsets of these symbols are intensified (i.e. highlighted) usually in pseudorandom order to produce an odd ball paradigm to induce p300 responses. 
```

Citation:
```
Farwell, L. A., & Donchin, E. (1988). Talking off the top of your head: toward a mental prosthesis utilizing event-related brain potentials. Electroencephalography and clinical Neurophysiology, 70(6), 510-523.

Ahani A, Moghadamfalahi M, Erdogmus D. Language-Model Assisted And Icon-based Communication Through a Brain Computer Interface With Different Presentation Paradigms. IEEE Trans Neural Syst Rehabil Eng. 2018 Jul 25. doi: 10.1109/TNSRE.2018.2859432.
```

## Demo
--------

All major functions and modules have demo and test files associated with them which may be run locally. This should help orient you to the functionality as well as serve as documentation. *If you add to the repo, you should be adding tests and fixing any test that fail when you change the code.*

For example, you may run the main BciPy demo by:

`python demo/bci_main_demo.py`

This demo will load in parameters and execute a demo task defined in the file. There are demo files for all modules listed above except helpers and utils. Run them as a python script!


## Testing
----------

When writing tests, put them in the correct module, in a tests folder, and prefix the file and test itself with `test_` in order for pytest to discover it. See other module tests for examples!

Development requirements must be installed before running: `pip install -r dev_requirements.txt`

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
coverage run --branch --source=bcipy -m pytest --mpl -k "not slow"

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
----------

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

### Contributions Welcome!

If you want to be added to the development team slack or have additional questions, please reach out to us at support@cambi.tech!

### Contribution Guidelines

We follow and will enforce the contributor's covenant to foster a safe and inclusive environment for this open source software, please reference this link for more information: https://www.contributor-covenant.org/

Other guidelines:
- All features require tests and a demo.
- All tests must pass to merge, even if they are seemingly unrelated to your work.
- Use Spaces, not Tabs.
- Use informative names for functions and classes.
- Document the input and output of your functions / classes in the code. eg in-line commenting and typing.
- Do not push IDE or other local configuration files.
- All new modules or major functionality should be documented outside of the code with a README.md.
	See README.md in repo or go to this site for inspiration: https://github.com/matiassingers/awesome-readme. Always use a Markdown interpreter before pushing. There are many free online or your IDE may come with one.

See this resource for examples: http://docs.python-guide.org/en/latest/writing/style/

## Contributors
---------------

All contributions are greatly appreciated!

[![image of contributors generated by https://contributors-img.web.app/ pulling from https://github.com/CAMBI-tech/BciPy/graphs/contributors](https://contrib.rocks/image?repo=CAMBI-tech/BciPy)](https://github.com/CAMBI-tech/BciPy/graphs/contributors)
