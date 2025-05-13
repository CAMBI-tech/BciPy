# BciPy: Brain-Computer Interface Software in Python

[![BciPy](https://github.com/CAMBI-tech/BciPy/actions/workflows/main.yml/badge.svg)](https://github.com/CAMBI-tech/BciPy/actions/workflows/main.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/96e31da4b0554dae9db7a1356556b0d5)](https://app.codacy.com/gh/CAMBI-tech/BciPy/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/CAMBI-tech/BciPy/fork)
[![Follow on Twitter](https://img.shields.io/twitter/follow/cambi_tech?label=Follow&style=social)](https://twitter.com/cambi_tech)


![CAMBI_logo](./bcipy/static/images/gui/CAMBI_full_logo.png) 

BciPy is a library for conducting Brain-Computer Interface experiments in Python. It is designed to be modular and extensible, allowing researchers to easily add new paradigms, models, and processing methods. The focus of BciPy is on paradigms for communication and control, including Rapid Serial Visual Presentation (RSVP) and Matrix Speller. 


See our official BciPy documentation including affiliations and more context information [here](https://bcipy.github.io/).

BciPy is released open-source under the BSD-3 clause. Please refer to [LICENSE.md](LICENSE.md)

*Please cite us when using!*

```text
Memmott, T., Koçanaoğulları, A., Lawhead, M., Klee, D., Dudy, S., Fried-Oken, M., & Oken, B. (2021). BciPy: brain–computer interface software in Python. Brain-Computer Interfaces, 1-18.
```

## Table of Contents

- [Dependencies](#dependencies)
  - [Linux](#linux)
  - [Windows](#windows)
  - [Mac](#mac)
- [Installation](#installation)
  - [BciPy Setup](#bcipy-setup)
  - [Editable Install and GUI usage](#editable-install-and-gui-usage)
  - [PyPi Install](#pypi-install)
  - [Make install](#make-install)
- [Usage](#usage)
  - [Package Usage](#package-usage)
  - [GUI Usage](#gui-usage)
  - [Client Usage](#client-usage)
    - [General Usage](#general-usage)
    - [Running Experiments or Tasks via Command Line](#running-experiments-or-tasks-via-command-line)
    - [Train a Signal Model via Command Line](#train-a-signal-model-via-command-line)
    - [Visualize ERP data from a session with Target / Non-Target labels via Command Line](#visualize-erp-data-from-a-session-with-target--non-target-labels-via-command-line)
  - [BciPy Simulator](#bcipy-simulator)
    - [Running the Simulator](#running-the-simulator)
- [Core Modules](#core-modules)
- [Paradigms](#paradigms)
  - [RSVPKeyboard](#rsvpkeyboard)
  - [Matrix Speller](#matrix-speller)
- [Glossary](#glossary)
- [Offset Determination and Correction](#offset-determination-and-correction)
  - [What is a Static Offset?](#what-is-a-static-offset)
  - [How to Determine the Offset](#how-to-determine-the-offset)
  - [Running Offset Determination](#running-offset-determination)
  - [Applying the Offset Correction](#applying-the-offset-correction)
  - [Using Make for Offset Determination](#using-make-for-offset-determination)
  - [Additional Resources](#additional-resources)
- [Contributions Welcome](#contributions-welcome)
  - [Contribution Guidelines](#contribution-guidelines)
  - [Contributors](#contributors)


## Dependencies

This project requires Python 3.9 or 3.10. 

It will run on the latest windows (10, 11), linux (ubuntu 22.04) and macos (Sonoma). Other versions may work as well, but are not guaranteed. To see supported versions and operating systems as of this release see our GitHub builds: [BciPy Builds](https://github.com/CAMBI-tech/BciPy/actions/workflows/main.yml). Please see notes below for additional OS specific dependencies before installation can be completed and reference our documentation here: <https://bcipy.github.io/hardware-os-config/>

### Linux

You will need to install the prerequisites defined in `scripts\shell\linux_requirements.sh` as well as `pip install attrdict3`.

### Windows

If you are using a Windows machine, you will need to install the [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).


### Mac

If you are using a Mac, you will need to install XCode and enable command line tools. `xcode-select --install`. If using an m1/2 chip, you may need to use the install script in `scripts/shell/m2chip_install.sh` to install the prerequisites. You may also need to use the Rosetta terminal to run the install script, but this has not been necessary in our testing using m2 chips.

If using zsh, instead of bash, you may encounter a segementation fault when running BciPy. This is due to an issue in a dependeancy of psychopy with no known fix as of yet. Please use bash instead of zsh for now.

## Installation

### BciPy Setup

In order to run BciPy on your computer, after ensuring the OS dependencies above are met, you can proceed to install the BciPy package.

### Editable Install and GUI usage
If wanting to run the GUI or make changes to the code, you will need to install BciPy in editable mode. This will ensure that all dependencies are installed and the package is linked to your local directory. This will allow you to make changes to the code and see them reflected in your local installation without needing to reinstall the package.

1. Git clone <https://github.com/BciPy/BciPy.git>.
2. Change directory in your terminal to the repo directory.
3. Install BciPy in development mode.
  ```sh
    pip install -e .
  ```

### PyPi Install
If you do not want to run the GUI or make changes to the code, you can install BciPy from PyPi. This will install the package and all dependencies, but will not link it to your local directory. This means that any changes you make to the code will not be reflected in your local installation. This is the recommended installation method if wanting to use the modules without making changes to the BciPy code.

```sh
pip install bcipy
```

### Make install

Alternately, if [Make](http://www.mingw.org/) is installed, you may run the follow command to install:

```sh
# install in development mode with all testing and demo dependencies
make dev-install
```

## Usage

The BciPy package may be used in two ways: via the command line interface (CLI) or via the graphical user interface (GUI). The CLI is useful for running experiments, training models, and visualizing data without needing to run the GUI. The GUI is useful for running experiments, editing parameters and training models with a more user-friendly interface.

### Package Usage

To run the package, you will need to import the modules you want to use. For example, to run the the system info module, you can run the following:


```python
from bcipy.helpers import system_utils
system_utils.get_system_info()
```

### GUI Usage

Run the following command in your terminal to start the BciPy GUI:

```sh
python bcipy/gui/BCInterface.py
```

Alternately, if Make is installed, you may run the follow command to start the GUI from the BciPy root directory:

```sh
make bci-gui
```

### Client Usage

Once BciPy is installed, it can be used via the command line interface. This is useful for running experiments, training models, and visualizing data without needing to run the GUI.

#### General Usage
Use the help flag to explore all available options:
```sh
bcipy --help
```

#### Running Experiments or Tasks via Command Line

You can invoke an experiment protocol or task directly using the `bcipy` command-line utility. This allows for flexible execution of tasks with various configurations.

##### Options

```sh
# Run with a User ID and Task
bcipy --user "bci_user" --task "RSVP Calibration"

# Run with a User ID and Experiment Protocol
bcipy --user "bci_user" --experiment "default"

# Run with Simulated Data
bcipy --fake

# Run without Visualizations
bcipy --noviz

# Run with Alerts after Task Execution
bcipy --alert

# Run with Custom Parameters
bcipy --parameters "path/to/valid/parameters.json"
```

These options provide flexibility for running experiments tailored to your specific needs.
  
#### Train a Signal Model via Command Line

To train a signal model (e.g., `PCARDAKDE` or `GazeModels`), use the `bcipy-train` command.
##### Basic Command
```sh
bcipy-train --help
```

##### Options
```sh
# Train using data from a specific folder
bcipy-train -d path/to/data

# Display data visualizations (e.g., ERPs)
bcipy-train -v

# Save visualizations to a file without displaying them
bcipy-train -s

# Train with balanced accuracy metrics
bcipy-train --balanced-acc

# Receive alerts after each task execution
bcipy-train --alert

# Use a custom parameters file
bcipy-train -p path/to/parameters.json
```
  
#### Visualize ERP data from a session with Target / Non-Target labels via Command Line

To visualize ERP data from a session with Target / Non-Target labels, use the `bcipy-erp-viz` command. This command allows you to visualize the data collected during a session and provides options for saving or displaying the visualizations.

##### Basic Command
```sh
bcipy-erp-viz --help
```

##### Options
```sh
# Run without a window prompt for a data session folder
bcipy-erp-viz -s path/to/data

# Run with data visualizations (ERPs, etc.)
bcipy-erp-viz --show

# Run with data visualizations that do not show, but save to file
bcipy-erp-viz --save

# Run with custom parameters (default is in bcipy/parameters/parameters.json)
bcipy-erp-viz -p "path/to/valid/parameters.json"
```

### BciPy Simulator
#### Running the Simulator

The simulator can be executed using the `bcipy-sim` command-line utility. 

##### Basic Command
```sh
bcipy-sim --help
```

##### Options
- `-d`: Path to the data folder.
- `-p`: Path to the custom parameters file.
- `-m`: Path to the trained model directory.
- `-n`: Number of iterations to run.

```sh
# Run the simulator with a specific data folder
bcipy-sim -d path/to/data
# Run the simulator with a custom parameters file
bcipy-sim -p path/to/parameters.json
# Run the simulator with a trained model directory
bcipy-sim -m path/to/model.pkl
# Run the simulator for a specific number of iterations
bcipy-sim -n 100
```

More comprehensive information can be found in the [Simulator Module README](./bcipy/simulator/README.md).


## Core Modules

### Top-Level Modules Overview

Each module includes its own README, demo, and tests. Click on the module name to view its README for more information.

#### [`Acquisition`](./bcipy/acquisition/README.md)
Captures data, returns desired time series, and saves to file at the end of a session.

#### [`Core`](./bcipy/core/README.md)
Core data structures and methods essential for BciPy operation.
  - Includes triggers, parameters, and raw data handling.

#### [`Display`](./bcipy/display/README.md)
Manages the display of stimuli on the screen and records stimuli timing.

#### [`Feedback`](./bcipy/feedback/README.md)
Provides feedback mechanisms for sound and visual stimuli.

#### [`GUI`](./bcipy/gui/README.md)
End-user interface for registered BCI tasks and parameter editing.
  - Key files: [BCInterface.py](./bcipy/gui/BCInterface.py) and [ParamsForm](./bcipy/gui/parameters/params_form.py).

#### [`Helpers`](./bcipy/helpers/README.md)
Utility functions for interactions between modules and general-purpose tasks.

#### [`IO`](./bcipy/io/README.md)
Handles data file operations such as loading, saving, and format conversion.
  - Supported formats: BIDS, BrainVision, EDF, MNE, CSV, JSON, etc.

#### [`Language`](./bcipy/language/README.md)
Provides symbol probability predictions during typing tasks.

#### [`Signal`](./bcipy/signal/README.md)
Includes EEG signal models, gaze signal models, filters, processing tools, evaluators, and viewers.

#### [`Simulator`](./bcipy/simulator/README.md)
Supports running simulations based on previously collected data.

#### [`Task`](./bcipy/task/README.md)
Implements user tasks and actions for BCI experiments.
  - Examples: RSVP Calibration, InterTaskAction.

### Entry Point and Configuration Modules

#### [`main.py`](./bcipy/main.py)
The main executor of experiments and the primary entry point into the application. See the [Running Experiments](#running-experiments-or-tasks-via-command-line) section for more information.

#### [`parameters/`](./bcipy/parameters/)
Contains JSON configuration files:
- [`parameters.json`](./bcipy/parameters/parameters.json): Main experiment and application configuration.
- [`device.json`](./bcipy/parameters/device.json): Device registry and configuration.
- [`experiments.json`](./bcipy/parameters/experiment/experiments.json): Experiment / protocol registry and configuration.
- [`phrases.json`](./bcipy/parameters/experiment/phrases.json): Phrase registry and configuration. This can be used to define a list of phrases used in the RSVP and Matrix Speller Copy phrase tasks. If not defined in parameters.json, the `task_text` parameter will be used.

#### [`config.py`](./bcipy/config.py)
Holds configuration parameters for BciPy, including paths and default data filenames.

#### [`static/`](./bcipy/static/)
Includes resources such as:
- Image and sound stimuli.
- Miscellaneous manuals and readable texts for the GUI.

## Paradigms

See the [Task README](./bcipy/task/README.md) for more information on all supported paradigms, tasks, actions and modes. The major paradigms are listed below.

### RSVPKeyboard

*RSVP KeyboardTM* is an EEG (electroencephalography) based BCI (brain computer interface) typing system. It utilizes a visual presentation technique called rapid serial visual presentation (RSVP). In RSVP, the options are presented rapidly at a single location with a temporal separation. Similarly in RSVP KeyboardTM, the symbols (the letters and additional symbols) are shown at the center of screen. When the subject wants to select a symbol, they await the intended symbol during the presentation and elicit a p300 response to a target symbol.


```text
Orhan, U., Hild, K. E., 2nd, Erdogmus, D., Roark, B., Oken, B., & Fried-Oken, M. (2012). RSVP Keyboard: An EEG Based Typing Interface. Proceedings of the ... IEEE International Conference on Acoustics, Speech, and Signal Processing. ICASSP (Conference), 10.1109/ICASSP.2012.6287966. https://doi.org/10.1109/ICASSP.2012.6287966
```

### Matrix Speller

Matrix Speller is an EEG (electroencephalography) based BCI (brain computer interface) typing system. It utilizes a visual presentation technique called Single Character Presentation (SCP). In matrix speller, the symbols are arranged in a matrix with fixed number of rows and columns. Using SCP, subsets of these symbols are intensified (i.e. highlighted) usually in pseudorandom order to produce an odd ball paradigm to induce p300 responses.

```text
Farwell, L. A., & Donchin, E. (1988). Talking off the top of your head: toward a mental prosthesis utilizing event-related brain potentials. Electroencephalography and clinical Neurophysiology, 70(6), 510-523.

Ahani A, Moghadamfalahi M, Erdogmus D. Language-Model Assisted And Icon-based Communication Through a Brain Computer Interface With Different Presentation Paradigms. IEEE Trans Neural Syst Rehabil Eng. 2018 Jul 25. doi: 10.1109/TNSRE.2018.2859432.
```

## Offset Determination and Correction

> [!CAUTION] System Synchronization
> Static offset determination and correction are critical steps before starting an experiment. BciPy uses LSL to acquire EEG data and Psychopy to present stimuli. The synchronization between the two systems is crucial for accurate data collection and analysis.

#### What is a Static Offset?

A static offset is the regular time difference between signals and stimuli presentation. This offset is determined through testing using a photodiode or another triggering mechanism. Once determined, the offset is corrected by shifting the EEG signal using the `static_offset` parameter in devices.json.

#### How to Determine the Offset

To determine the static offset, you can run a timing verification task (e.g., `RSVPTimingVerification`) with a photodiode attached to the display and connected to your device. After collecting the data, use the `offset` module to analyze the results and recommend an offset correction value.

#### Running Offset Determination

To calculate the offset and display the results, use the following command:

```bash
python bcipy/helpers/offset.py -r
```

This will analyze the data and recommend an offset correction value, which will be displayed in the terminal.

#### Applying the Offset Correction

Once you have the recommended offset value, you can apply it to verify system stability and display the results. For example, if the recommended offset value is `0.1`, run the following command:

```bash
python bcipy/helpers/offset.py --offset "0.1" -p
```

#### Using Make for Offset Determination

If `Make` is installed, you can simplify the process by running the following command to determine the offset and display the results:

```sh
make offset-recommend
```

#### Additional Resources

For more information on synchronization and timing, refer to the following documentation:
- [LSL Synchronization Documentation](https://labstreaminglayer.readthedocs.io/info/time_synchronization.html)
- [PsychoPy Timing Documentation](https://www.psychopy.org/general/timing/index.html)


## Glossary

***Stimuli***: A single letter, tone or image shown (generally in an inquiry). Singular = stimulus, plural = stimuli.

***Trial***: A collection of data after a stimuli is shown. A----

***Inquiry***: The set of stimuli after a fixation cross in a spelling task to gather user intent. A ---- B --- C ----

***Series***: Each series contains at least one inquiry. A letter/icon decision is made after a series in a spelling task.

***Session***: Data collected for a task. Comprised of metadata about the task and a list of Series.

***Protocol***: A collection of tasks and actions to be executed in a session. This is defined for each experiment and can be registered in experiments.json via the BCI GUI.

***Experiment***: A protocol with a set of parameters. This is defined within experiments and can be registered in experiments.json via the BCI GUI.

***Task***: An experimental design with stimuli, trials, inquiries and series for use in BCI. For instance, "RSVP Calibration" is a task.

***Action***: A task without a paradigm. For instance, "RSVP Calibration" is a task, but "InterTaskAction" is an action. These are most often used to define the actions that take place in between tasks.

***Mode***: Common design elements between task types. For instance, Calibration and Free Spelling are modes.

***Paradigm***: Display paradigm with unique properties and modes. Ex. Rapid-Serial Visual Presentation (RSVP), Matrix Speller, Steady-State Visual Evoked Potential (SSVEP).

## Contributions Welcome

If you want to be added to the development team slack or have additional questions, please reach out to us at <support@cambi.tech>!

### Contribution Guidelines

We follow and will enforce the contributor's covenant to foster a safe and inclusive environment for this open source software, please reference this link for more information: <https://www.contributor-covenant.org/>

We welcome all contributions to BciPy! Please follow the guidelines below:

- All modules require tests and a demo.
- All tests must pass to merge, even if they are seemingly unrelated to your work.
- Use Spaces, not Tabs.
- Use informative names for functions and classes.
- Document the input and output of your functions / classes in the code. eg in-line commenting and typing.
- Do not push IDE or other local configuration files.
- All new modules or major functionality should be documented outside of the code with a README.md.
-- See README.md in repo or go to this site for inspiration: <https://github.com/matiassingers/awesome-readme>. Always use a Markdown interpreter before pushing.

See this resource for examples: <http://docs.python-guide.org/en/latest/writing/style/>

### Contributors

All contributions are greatly appreciated!

[![image of contributors generated by https://contributors-img.web.app/ pulling from https://github.com/CAMBI-tech/BciPy/graphs/contributors](https://contrib.rocks/image?repo=CAMBI-tech/BciPy)](https://github.com/CAMBI-tech/BciPy/graphs/contributors)
