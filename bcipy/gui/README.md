# RSVP Keyboard GUI
======================================

This module contains all GUI code used in BciPy. The base window class, BCIGui, is contained in gui_main.py, and contains methods for easily adding widgets to a given window. BCInterface.py launches the main GUI window. There are also interfaces for collecting and editing data (parameters and field data for experiments.)

## Dependencies
-------------
This project was written in PyQt6.

## Project structure
---------------
Name | Description
------------- | -------------
BCInterface.py | Defines main GUI window. Selection of user, experiment and task.
gui_main.py | BCIGui containing methods for adding buttons, images, etc. to GUI window
parameters/params_form.py | Defines window for setting BCInterface parameters
experiments/ExperimentRegistry.py | GUI for creating new experiments to select in BCInterface.
experiments/FieldRegistry.py | GUI for creating new fields for experiment data collection.
experiments/ExperimentField.py | GUI for collecting a registered experiment's field data.


The folder 'bcipy/static/images/gui' contains images for the GUI.
Parameters loaded by BCInterface parameter definition form can be found in 'bcipy/parameters/parameters.json'.

To run the GUI, do so from the root, as follows:  

`python bcipy/gui/BCInterface.py`  

Contributors:

- Tab Memmott
- Matthew Lawhead
- Carson Reader
- Dani Smektala
