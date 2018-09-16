# RSVP Keyboard GUI
======================================

This is the GUI for BciPy. The base window class, BCIGui, is contained in gui_main.py, and contains methods for easily adding wxPython widgets to a given window. BCInterface.py launches the main GUI window. 

## Features
-----------

-Buttons, text boxes, drop-down menus, and other GUI elements are easy to add  
-Read/write framework for JSON files  

## Dependencies
-------------
This project was written in wxPython version 4.0.0.  
wxPython is a dependency of Psychopy.

## Project structure
---------------
Name | Description
------------- | -------------
BCInterface.py | Defines main GUI window
gui_main.py | BCIGui class extending wx.Frame, containing methods for adding buttons, images, etc. to GUI window
mode/RSVPKeyboard.py | Defines window for launching RSVPKeyboard
params_form.py | Defines window for setting RSVPKeyboard parameters


The folder 'bcipy/static/images/gui_images' contains images for the GUI.
Parameters loaded by the RSVPKeyboard parameter definition form can be found in 'bcipy/parameters/parameters.json'.

To run the GUI from this directory:  

`python BCInterface.py`  


Initially written by. Dani Smektala under the supervision of Tab Memmott @ OHSU
