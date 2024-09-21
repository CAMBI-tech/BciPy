## RSVP Simulator

### Overview

This Simulator module aims to automate experimentation by sampling EEG data from prior sessions and running given models in RSVP task loop, thus simulating a live session.

### Run steps

`main.py` is the entry point for program. After following `BciPy` readme steps for setup, run the module from terminal:

```
(venv) $ python bcipy/simulator -h
usage: simulator [-h] -d DATA_FOLDER [-g GLOB_PATTERN] -m MODEL_PATH -p PARAMETERS [-n N]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_FOLDER, --data_folder DATA_FOLDER
                        Raw data folders to be processed.
  -g GLOB_PATTERN, --glob_pattern GLOB_PATTERN
                        glob pattern to select a subset of data folders Ex. "*RSVP_Copy_Phrase*"
  -m MODEL_PATH, --model_path MODEL_PATH
                        Signal models to be used
  -p PARAMETERS, --parameters PARAMETERS
                        Parameter File to be used
  -n N                  Number of times to run the simulation
```

#### Program Args

- `d` : the data wrapper folder argument is necessary. This folder is expected to contain 1 or more session folders. Each session folder should contain
  _raw_data.csv_, _triggers.txt_, _parameters.json_. These files will be used to construct data pool from which simulator will sample EEG responses. 
- `g` : optional glob filter that can be used to select a subset of data.
  - Ex. `"*Matrix_Copy*Jan_2024*"` will select all data for all Matrix Copy Phrase sessions recorded in January of 2024 (assuming the BciPy folder naming convention).
  - Glob patterns can also include nested directories (ex. `"*/*Matrix_Copy*"`).
- `p` : path to the parameters.json file used to run the simulation. These parameters will be applied to
  all raw_data files when loading.
- `m`: all pickle (.pkl) files in this directory will be loaded as signal models.

#### Sim Output Details

Output folders are generally located in the `simulator/generated` directory. Each simulation will create a new directory. The directory name will be  prefixed with `SIM` and will include the current date and time.

- `parameters.json` captures params used for the simulation
- `sim.log` is a log file for the simulation

A directory is created for each simulation run. The directory contents are similar to the session output in a normal bcipy task. Each run directory contains:

- `run_n.log` log file specific to the run
- `session.json` session data output for the task, including evidence generated for each inquiry and overall metrics.
- `session.xlsx` session data summarized in an excel spreadsheet with charts for easier visualization.
