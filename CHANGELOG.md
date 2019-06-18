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