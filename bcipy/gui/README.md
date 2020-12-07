# RSVP Keyboard GUI
======================================

This is the GUI for BciPy. The base window class, BCIGui, is contained in gui_main.py, and contains methods for easily adding widgets to a given window. BCInterface.py launches the main GUI window. 

## Features
-----------

-Buttons, text boxes, drop-down menus, and other GUI elements are easy to add  
-Read/write framework for JSON files  

## Dependencies
-------------
This project was written in wxPython version 4.0.0 and PyQt5. We are deprecating the wxPython UIs in future releases.

## Project structure
---------------
Name | Description
------------- | -------------
BCInterface.py | Defines main GUI window. Selection of user, experiment and task.
gui_main.py | BCIGui containing methods for adding buttons, images, etc. to GUI window
params_form.py | Defines window for setting BCInterface parameters
experiments/ExperimentRegistry.py | GUI for creating new experiments to select in BCInterface.


The folder 'bcipy/static/images/gui_images' contains images for the GUI.
Parameters loaded by BCInterface parameter definition form can be found in 'bcipy/parameters/parameters.json'.

To run the GUI, do so from the root, as follows:  

`python bcipy/gui/BCInterface.py`  

Contributors:

- Tab Memmott
- Matthew Lawhead
- Dani Smektala
