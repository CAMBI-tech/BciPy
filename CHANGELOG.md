# 2.0.0

## Contributions

This version contains major refactoring efforts and features. We anticipate a few additional refactor efforts in the near term based on feature requests from the community and CAMBI. These will support multi-modality, data sharing, and more complex language modeling.  

### Added

- `run-with-defaults`: make command for running `bcipy`

### Updated

- `LICENSE.md`: to used the Hippocratic license 2.1
- `CODE_OF_CONDUCT.md`: to latest version of the Contributor Covenant
- `bcipy.main`: formally, `bci_main`. To give a better console entry point and infrastructure for integration testing. In the terminal, you can now run `bcipy` instead of `python bci_main.py` 

### Removed


# 1.5.0

## Contributions

This version contains major refactoring and tooling improvements across the codebase. In addition, it introduces the concept of BciPy Experiments and Fields. Below we describe the major changes along with a PR# in github where applicable. 

### Added
- Language model histogram #91 
- BciPy official glossary (Sequence -> Inquiry & Epoch -> Series) #121 
- System information to `system_utils` (cpu, platform, etc) #98 
- BciPy Experiments and Fields: See PRs #113 #111 and #114 for more information on the additions!
- `.bcipy` system directory to support experiment and fields #100 
- support for python 3.7
- `rsvp/query_mechanisms`: to model the way we build inquiries #108 
- `Makefile`: contains useful install and development commands
- `convert`: a module for data conversions that will be useful for data sharing. Implemented a conversion function to the EDF format. #104 
- `exceptions`: a module for BciPy core exceptions

### Updated
- `acquisition`: refactored the acquisition module to separate the concept of a device (ex. DSI-24 headset) and a connection method to that device (TCP or LSL). #122 
- `setup.py`: with new repo location and CAMBI official support email 
- `offline_analysis`: to pull parameters from session file #90 
- `requirements.txt`: to the latest available #99 #107 
- `Parameters` (added help text, removed redundant parameters). Refactored to make them immutable. #101 
- `gui_main `: to use PyQt5. We will refactor all GUI code to use this in the future. After this PR, the signal viewer (WxPython) and a couple of loading functions will remain (Tk). #102 
- `BCInterface` : updated to use new gui_main methods. Added user if validations. #102  #120 
- `params_form`: moved into a parameters modules within GUI and updated to use PyQt5. #109 
- `dev_requirements`: used to be called test_requirements. It contains more than that, so we updated the name! #99 
- `README`: with relevant updates and contributors 

### Removed
- `RSVPKeyboard.py`


# 1.4.2

## Contributions

### Added

- Artifact Rejection
    - `signal.evaluate.Evaluator`
        - Evaluates sequences and flags sequences that break rules
    - `signal.evaluate.Rule`
        - Defines rules that sequences should follow
- Re-added progress indicator for offline analysis
- Scripts
    - `timing_tools.py` to help test display timing before experiments
- Tests
- Linting
- Documentation improvements

### Updated

- DAQ file writing. Instead of writing to `raw_data.csv` during task executions, optionally write `raw_data.csv` on call to `stop_acquisition`, and only write to SQLite database during task.
- `get_data` queries in `buffer_server` to increase speed on Windows machines. 
- RSVP sequence stimuli presentation. Reduces reported timing slips on Windows and Linux machines. 
    - List of stimuli to be presented is now generated before each sequence is presented, rather than generating stimuli during the sequence.
    - The screen is now only drawn once per stimulus, rather than redrawing the screen every frame.
- Signal viewer to shut down when data is no longer streaming
- `main_frame` and `copy_phrase` to fix bug that prevents copy phrase tasks from completing in fake data mode
- Target generation for icon-matching tasks has been changed to minimize instances of duplicate target stimuli over the course of an experiment
- Fixed issue with icon-matching tasks flashing to desktop on Windows machines
- `params_form.py` is now launched by creating another wxPython frame, rather than launching a subprocess

## Removed

- Module additions via `__init__`

# 1.4.1

Patch for gui.viewer module. Missing init file.

# 1.4.0

This release focused on bug fixes, exposing parameters, and refactoring. Further dual screen configuration and testing was done to allow for simultaneous signal viewing and task operation. 

## Added

    - Dual screen configuration / updated support
    - Parameters:
            - Copy phrase decision thresholds
            - Inter-sequence Feedback level thresholds
   

## Updated
    - RSVP Display: refactor 
    - Decision Maker / Evidence Fusion: refactor
    - Signal Viewer: more distinct channel names
    - bci_main: shutdown handling and bug fix
    - Language Model Helper: bug fix for negative probabilities

## Removed
    - Multicolor Text
    - Old LSL viewer


# 1.3.0

This release focused on the addition of a Signal Viewer, Inter-sequence Feedback Task, Signal Processing / Decomposition Methods, and miscellaneous cleanup. 

## Added

    - PSD: Power spectral density methods
    - DSI VR-300 integration
    - Logging session configuration and setup
    - Version and git commit extraction
    - Inter-sequence feedback task
    - Backspace frequency parameterization and implementation
    - Bandpass and Notch Filter implementation
    - Custom Exceptions
   

## Updated
    - Refactor RSVP Task: Icon-to-Icon (WIP)
    - Refactored Signal Module
    - Dependencies (PsychoPy, pylsl, numpy, pandas, WxPython)
    - Tests
    - Documentation

## Removed
    - Bar Graph implementation


# 1.2.0

This release focused on the addition of a new Alert Tone Task, integration of the Language Model, and other fixes to make Copy Phrase work better.

## New Features
    - Alert Tone Task: a new calibration task that plays an alert tone prior to displaying the presentation letter. Adds parameters for the directory in which sounds (.wav files) are located as well as the length of the delay.
    - SPACE symbol customization: allows users to customize the display of the SPACE character.
    - New Language Model: experimental; adds a new oclm Language Model and allows users to select between this and the standard prelm model. Note that this functionality is still incomplete.

## Updates
    - Language Model integration: the Language Model now correctly starts up a Docker container on both Windows and Mac machines.
    - Fixes to the CopyPhraseWrapper in its usage of the Language Model, including converting its output to the probability domain for integration, and correctly handling space characters.
    - Fix for Copy Phrase backspace selection.
    - Fixed issue loading EEG Classifier pkl files.
    - General code cleanup in the acquisition module.
    - Simplified code for starting a new task.
    - There is now a task_registry which lists all available tasks and allows code to enumerate them.
    - More obvious feedback when a CopyPhrase letter has been typed.
    - Users can now override configuration for parameters with a drop-down menu. The previous behavior was to restrict users to the suggested values.


# 1.1.0

This is a working version for use with LSL on Calibration and Copy phrase tasks. It will break implementation from previous release. Future minor versions should not do this.

## Updated:
- Structure of Signal Module
- Signal Model and Trial Reshaper Refactored
- Documentation
- Data acquisition client naming

## Added:
- Image scaling
- Initial Icon Word matching task
- Task breaks
- LSL offset correction
- Unittests
- AUC printing to filename
- logging
- Initial signal viewer
- Initial average ERP generation plots
- GUI enhancements

## Removed
- Duplicate dependencies
