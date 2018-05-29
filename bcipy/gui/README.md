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
This project was written in wxPython version 4.0.0a3, and pyglet version 1.3.0b1.  
Both wxPython and pyglet are dependencies of Psychopy.

## Project structure
---------------
Name | Description
------------- | -------------
utility/gui_fx.py  | All GUI execution code
utility/parameters.json  | Parameters file containing all parameter names, default values, suggested values, etc.
RSVPKeyboard.py | Registration of all GUI elements


The 'static' folder contains images.

To run the GUI:  

`python RSVPKeyboard.py`  


Initially written by. Dani Smektala under the supervision of Tab Memmott @ OHSU

