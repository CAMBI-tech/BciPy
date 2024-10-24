# BciPy: Brain-Computer Interface Software in Python


[![BciPy](https://github.com/CAMBI-tech/BciPy/actions/workflows/main.yml/badge.svg)](https://github.com/CAMBI-tech/BciPy/actions/workflows/main.yml)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/CAMBI-tech/BciPy/fork)
[![Follow on Twitter](https://img.shields.io/twitter/follow/cambi_tech?label=Follow&style=social)](https://twitter.com/cambi_tech)


BciPy is a library for conducting Brain-Computer Interface experiments in Python. It functions as a standalone application for experimental data collection or you can take the tools you need and start coding your own system. See our official BciPy documentation including affiliations and more context information [here](https://bcipy.github.io/).

It will run on the latest windows (7, 10, 11), linux (ubuntu 22.04) and macos (Big Sur). Other versions may work as well, but are not guaranteed. To see supported versions and operating systems as of this release see here: [BciPy Builds](https://github.com/CAMBI-tech/BciPy/actions/workflows/main.yml).

*Please cite us when using!*

```
Memmott, T., Koçanaoğulları, A., Lawhead, M., Klee, D., Dudy, S., Fried-Oken, M., & Oken, B. (2021). BciPy: brain–computer interface software in Python. Brain-Computer Interfaces, 1-18.
```

## Dependencies
---------------
This project requires Python 3.8 or 3.9. Please see notes below for additional OS specific dependencies before installation can be completed and reference our documentation/FAQs for more information: https://bcipy.github.io/hardware-os-config/

### Linux

You will need to install the prerequisites defined in `scripts\shell\linux_requirements.sh` as well as `pip install attrdict3`.

### Windows

If you are using a Windows machine, you will need to install the [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

*python 3.9 only!*
You will need to install pyWinhook manually. See [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pywinhook) for the appropriate wheel file (`pyWinhook‑1.6.2‑cp39‑cp39‑win_amd64.whl`). Then run `pip install <path_to_wheel_file>`. We also include the 64-bit wheel file in the `.bcipy/downloads/` directory.

### Mac

If you are using a Mac, you will need to install XCode and enable command line tools. `xcode-select --install`. If using an m1/2 chip, you will need to use the install script in `scripts/shell/m2chip_install.sh` to install the prerequisites. You may also need to use the Rosetta terminal to run the install script, but this has not been necessary in our testing using m2 chips. 

If using zsh, instead of bash, you may encounter a segementation fault when running BciPy. This is due to an issue in a dependeancy of psychopy with no known fix as of yet. Please use bash instead of zsh for now. 

## Installation
---------------

#### BciPy Setup

In order to run BciPy on your computer, after following the dependencies above, you will need to install the BciPy package.

To install for use locally and use of the GUI:
1. Git clone https://github.com/BciPy/BciPy.git
2. Change directory in your terminal to the repo directory.
3. Install the kenlm language model package. `pip install kenlm==0.1 --global-option="--max_order=12"`.
4. Install PsychoPy with no dependencies. `pip install psychopy==2023.2.1 --no-deps`.
5. Install BciPy in development mode. `pip install -e .`


If wanting the latest version from PyPi and to build using modules:
1. `pip install bcipy`

Alternately, if [Make](http://www.mingw.org/) is installed, you may run the follow command to install:

```sh
# install in development mode
make dev-install
```

#### Client Usage
   	Invoke an experiment protocol or task directly using command line utility `bcipy`.
		- You can pass it attributes with flags, if desired.
				Running with a User ID and Task: `bcipy --user "bci_user" --task "RSVP Calibration"`
				Running with a User ID and Tasks with a registered Protocol: `bcipy --user "bci_user" --experiment "default"`
				Running with fake data: `bcipy --fake`
				Running without visualizations: `bcipy --noviz`
				Running with alerts after each Task execution: `bcipy --alert`
				Running with custom parameters: `bcipy --parameters "path/to/valid/parameters.json"`

		- Use the help flag to see other available input options: `bcipy --help`

#####  Example Usage as a Package

```python
from bcipy.helpers import system_utils
system_utils.get_system_info()
```

#### Example Usage through the GUI

Run the following command in your terminal to start the BciPy GUI:
```sh
python bcipy/gui/BCInterface.py
```

Alternately, if Make is installed, you may run the follow command to start the GUI from the BciPy root directory:

```sh
make bci-gui
```


#### Simulator Usage

The simulator can be run using the command line utility `bcipy-sim`.

Ex. 
`bcipy-sim -d my_data_folder/ -p my_parameters.json -m my_models/ -n 5`

Run `bcipy-sim --help` for documentation or see the README in the simulator module.


## Glossary
-----------

***Stimuli***: A single letter, tone or image shown (generally in an inquiry). Singular = stimulus, plural = stimuli.

***Trial***: A collection of data after a stimuli is shown. A----

***Inquiry***: The set of stimuli after a fixation cross in a spelling task to gather user intent. A ---- B --- C ----

***Series***: Each series contains at least one inquiry. A letter/icon decision is made after a series in a spelling task.

***Session***: Data collected for a task. Comprised of metadata about the task and a list of Series.

***Protocol***: A collection of tasks and actions to be executed in a session. This is defined as within experiments and can be registered using the BciPy GUI.

***Task***: An experimental design with stimuli, trials, inquiries and series for use in BCI. For instance, "RSVP Calibration" is a task.

***Mode***: Common design elements between task types. For instance, Calibration and Free Spelling are modes.

***Paradigm***: Display paradigm with unique properties and modes. Ex. Rapid-Serial Visual Presentation (RSVP), Matrix Speller, Steady-State Visual Evoked Potential (SSVEP).


## Core Modules
---------------

This a list of the major modules and their functionality. Each module will contain its own README, demo and tests. Please check them out for more information!

- `acquisition`: acquires data, gives back desired time series, saves to file at end of session.
- `display`: handles display of stimuli on screen and passes back stimuli timing.
- `signal`: eeg signal models, gaze signal models, filters, processing, evaluators and viewers.
- `gui`: end-user interface into registered bci tasks and parameter editing. See BCInterface.py.
- `helpers`: helpful functions needed for interactions between modules, basic I/O, and data visualization.
- `language`: gives probabilities of next symbols during typing.
- `parameters`: location of json parameters. This includes parameters.json (main experiment / app configuration) and device.json (device registry and configuration).
- `static`: image and sound stimuli, misc manuals, and readable texts for gui.
- `task`: bcipy implemented user tasks. Main collection of bci modules for use during various experimentation. Ex. RSVP Calibration.
- `feedback`: feedback mechanisms for sound and visual stimuli.
- `main`: executor of experiments. Main entry point into the application
- `config`: configuration parameters for the application, including paths and data filenames.
- `simulator`: provides support for running simulations based off of previously collected data.


## Paradigms
------------

See `bcipy/task/README.md` for more information on all supported paradigms and modes. The following are the supported and validated paradigms:


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

This demo will load in parameters and execute a demo task defined in the file. There are demo files contained in most modules, excepting gui, signal and parameters. Run them as a python script!


## Offset Determination and Correction
--------------------------------------

Static offset determination and correction are critical steps before starting an experiment. BciPy uses LSL to acquire EEG data and Psychopy to present stimuli. 

[LSL synchronization documentation](https://labstreaminglayer.readthedocs.io/info/time_synchronization.html)
[PsychoPy timing documentation](https://www.psychopy.org/general/timing/index.html)

A static offset is the regular time difference between our signals and stimuli. This offset is determined through testing via a photodiode or other triggering mechanism. The offset correction is done by shifting the EEG signal by the determined offset using the `static_offset` parameter. 

After running a timing verification task (such as, RSVPTimingVerification) with a photodiode attached to the display and connected to a device, the offset can be determined by analyzing the data. Use the `offset` module to recommend an offset correction value and display the results.

To run the offset determination and print the results, use the following command:

```bash
python bcipy/helpers/offset.py -r
```

After running the above command, the recommended offset correction value will be displayed in the terminal and can be passed to determine system stability and display the results.

```bash
# Let's say the recommneded offset value is 0.1
python bcipy/helpers/offset.py --offset "0.1" -p

```

Alternately, if Make is installed, you may run the follow command to run offset determination and display the results:

```sh
make offset-recommend
```

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

## Type Checking
----------------

This project enforces `mypy` type checking. The typing project configuration is found in the mypy.ini file. To run type checking, run the following command:

```sh
mypy bcipy
```

To generate a report, run the following command:

```sh
mypy --html-report bcipy
```

Alternately, if Make is installed, you may run the follow command to run mypy:

```sh
make type
```


### Contributions Welcome!

If you want to be added to the development team slack or have additional questions, please reach out to us at support@cambi.tech!

### Contribution Guidelines

We follow and will enforce the contributor's covenant to foster a safe and inclusive environment for this open source software, please reference this link for more information: https://www.contributor-covenant.org/

Other guidelines:
- All modules require tests and a demo.
- All tests must pass to merge, even if they are seemingly unrelated to your work.
- Use Spaces, not Tabs.
- Use informative names for functions and classes.
- Document the input and output of your functions / classes in the code. eg in-line commenting and typing.
- Do not push IDE or other local configuration files.
- All new modules or major functionality should be documented outside of the code with a README.md.
	See README.md in repo or go to this site for inspiration: https://github.com/matiassingers/awesome-readme. Always use a Markdown interpreter before pushing.

See this resource for examples: http://docs.python-guide.org/en/latest/writing/style/

## Contributors
---------------

All contributions are greatly appreciated!

[![image of contributors generated by https://contributors-img.web.app/ pulling from https://github.com/CAMBI-tech/BciPy/graphs/contributors](https://contrib.rocks/image?repo=CAMBI-tech/BciPy)](https://github.com/CAMBI-tech/BciPy/graphs/contributors)
