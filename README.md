# Brain- Computer Interface Codebase
======================================

This is the codebase for BCI suite. 

## Features
-----------

- RSVPKeyboard. 
	RSVP KeyboardTM is an EEG (electroencephalography) based BCI (brain
	computer interface) typing system. It utilizes a visual presentation technique
	called rapid serial visual presentation (RSVP). In RSVP, the options are
	presented rapidly at a single location with a temporal separation. Similarly
	in RSVP KeyboardTM, the symbols (the letters and additional symbols) are
	shown at the center of screen. When the subject wants to select a symbol,
	he/she awaits for the intended symbol during the presentation.

	To run on windows, execute the .exe to start with GUI. Otherwise, run RSVPKeyboard.py in gui/ to begin. 

- Shuffle Speller.
	TBD.

- Matrix Speller.
	TBD 

## Dependencies
-------------
This project requires Psychopy and Python v 2.7. See requirements.txt


## Installation
------------

# RSVP Keyboard Setup

In order to run **RSVP Keyboard** on your computer, first install **Python 2.7** [from here.](https://www.python.org/downloads/) Then, you need to install required modules for RSVP Keyboard. There are two methods for this, choose one of:


1. Run moduleLoader.py.


2. Use pip to iteratively install required modules.
    - pip install -r /path/to/requirements.txt

    After pip is done, download two modules that are left according to your OS(64 or 32 bit), which are numpy+mkl 1.13.1 [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy) and scipy 0.19.1 [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy).
    - pip install /path/to/numpy‑1.13.1+mkl‑cp27‑cp27m‑win_amd64.whl
    - pip install /path/to/scipy‑0.19.1‑cp27‑cp27m‑win_amd64.whl

You are ready to run RSVP Keyboard.

## Modules and Vital Functions
------------------------------

- acquistion: acquires data, gives back desired time series, saves at end of session
- display: handles display of stimuli on screen, passing back stimuli timing
- eeg_model: trains and classifies eeg responses based on eeg and triggers
- gui: end-user interface into system
- io: input/output functions
- language_model
- parameters
- static

- bci_main: executor of experiment. 


