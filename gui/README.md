# RSVP Keyboard GUI
======================================

This is the GUI for RSVP Keyboard. All GUI elements (buttons, input boxes, etc.) are registered in RSVPKeyboard.py, and gui_fx.py runs the program and handles user interaction.

## Features
-----------

-Buttons, text boxes, drop-down menus, windows, and other GUI elements are easy to add  
-Scroll bars  
-Read/write framework for JSON files  

## Dependencies
-------------
This project requires wxPython version 4.0.0a3, and pyglet version 1.3.0b1.  
Both wxPython and pyglet are dependencies of Psychopy.

## Project structure
---------------
Name | Description
------------- | -------------
convertinitojson.py | Converts ini files in 'static' folder to single JSON file.
gui_fx.py  | All GUI execution code
parameters.json  | Parameters file containing all parameter names, default values, suggested values, etc.
RSVPKeyboard.py | Registration of all GUI elements

The 'static' folder contains images and old ini files.

## Installation
------------

After downloading and unzipping the source, cd into the ohsu-rsvp-gui repo, then run:  

`pip install -r requirements.txt`  


To run the GUI:  

`python RSVPKeyboard.py`  

To run the included test file, cd into the 'testing' directory, then run:  
`py.test pytestfile.py`

## Source
----------

- Source Code: https://bitbucket.org/smektala/ohsu-rsvp-gui/src
- Issue Tracker: https://bitbucket.org/smektala/ohsu-rsvp-gui/issues
