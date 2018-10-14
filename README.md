# Brain- Computer Interface Codebase
------------------------------------

### What is it?

It is a Brain-computer interface framework written in Python. It can function as a standalone or you can take the tools you need and start coding your own / additional comonpents.

It should, based on our dependancies, work on most recent operating systems, however it has only been verified on Windows (10) and Mac OSx (High Sierra) at this time.

### Contributions Welcome!

This is our first release. It is verified using LSL with DSI and gtec for the Calibration modes only at this time with both image and text stimuli. It comes with a fake data server to help you develop while mocking real time EEG acquistion. We are taking all suggestions at this time for additions, eventually we'll make a contributions wishlist. If you want to be added to the development team, reach out to us and we'll add you to the team slack.

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

You must install Docker and Docker-Machine to use the Language Model developed by CSLU. There are instructions in the language model directory for getting the image you need (think of it as a callable server). If not using or rolling your own, set fake_lm to true in the parameters.json file.

To use all the goodies locally (including the GUI and demo scripts)
1. Git clone https://github.com/BciPy/BciPy.git
2. Change directory in your terminal to the repo
3. Run `pip install -e .`
4. If using Mac, you will need to install XCode and enable command line tools. `xcode-select --install`
5. If you're on Windows, you may need to uninstall pygame (`pip uninstall pygame`). Psychopy, for historical reasons, keeps pygame but it just spams your console logs if you only want to use pyglet (which we are in this repository!)

To just use the built-in functions:
1. `pip install bcipy`

## Usage Locally

Start by running `python bcipy/gui/BCInterface.py` in your command prompt or terminal. You may also invoke the experiment directly using command line tools for bci_main.py.

Ex.`python bci_main.py` *this will default parameters, mode, user, and types.*


You can pass it attributes with flags, if desired.

Ex. `python bci_main.py --user "bci_user" --mode "RSVP"`

## Example usage as a package

```python
from bcipy.helpers import system_utils
system_utils.get_system_info()
```

## Modules and Vital Functions
------------------------------

This a list of the major modules and their functionality. Each module will contain its own README, demo and test scripts. Please check them out for more information!

- `acquistion`: acquires data, gives back desired time series, saves at end of session.
- `display`: handles display of stimuli on screen, passing back stimuli timing.
- `signal_model`: trains and classifies eeg responses based on eeg and triggers.
- `gui`: end-user interface into system. See BCInterface.py and RSVPKeyboard.py.
- `helpers`: input/output functions needed for system, as well as helpful intilization functions.
- `utils`: utility functions needed for operation and installation.
- `language_model`: gives prob of letters during typing.
- `parameters`: json file for parameters.
- `static`: images, misc manuals, and readable texts for gui.
- `bci_main`: executor of experiments.

## Demo and Tests
-----------------

All major functions and modules have demo and test files associated with them which may be run locally. This should help orient you to the functionality as well as serve as documentation. *If you add to the repo, you should be adding tests and fixing any test that fail when you change the code.*

For example, you may run the bci_main demo by:

`python demo/bci_main_demo.py`

This demo will load in parameters and execute a demo task defined in the file. There are demo files for all modules listed above except language_model, helpers, and utils.

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
8. All new modules or major functionality should be documented outside of the code with a README.md. See REAME.md in repo or go to this site for inspiration: https://github.com/matiassingers/awesome-readme. Always use a Markdown interpreter before pushing. There are many free online or your IDE may come with one.

Use this resource for examples: http://docs.python-guide.org/en/latest/writing/style/

## Testing
----------

Please have all requirements installed. When writing tests, put them in the correct module, in a tests folder, and prefix the file and test itself with `test` in order for pytest to discover it. See other module's tests for examples!

To run all tests, in the command line:

```python
py.test
```


To run a single modules tests (ex. acquisition), in the command line:

```python
py.test acquisition
```

To generate test coverage metrics, in the command line:

```python
coverage run -m py.test

#Generate a command line report
coverage report

# Generate html doc in the bci folder. Navigate to index.html and click.
coverage html

```

## Authorship
--------------

Tab Memmott (OHSU)


Aziz Kocanaogullari (NEU)


Matthew Lawhead (OHSU- OCTRI)


Berkan Kadioglu (NEU)


Dani Smektala (OHSU)


Andac Demir (NEU)


Shaobin Xu (NEU)


Shiran Dudy (OHSU)
