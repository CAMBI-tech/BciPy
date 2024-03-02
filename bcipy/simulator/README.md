## RSVP Simulator

### Overview

This Simulator module aims to automate experimentation by sampling EEG data from prior sessions and
running given models
in RSVP task loop, thus simulating a
live session.

### Run steps

`main.py` is the entry point for program. After following, `BciPy` readme steps for setup, run the
module from shell as
such:

``python3 bcipy/simulator -d "pathToDataFolder" -p "pathToParametersFile" -sm "pathToSignalModel.pkl"``

Then view generated LOG file for analysis

#### Program Args

- `d` : the data folder argument is necessary. This data will be used to construct data pool from
  which simulator will
  sample EEG responses. Should contain
  _raw_data.csv_, _triggers.txt_, _parameters.json_
- `g` : optional glob filter that can be used to select a subset of data.
  - Ex. `"*Matrix_Copy*Jan_2024*"` will select all data for all Matrix Copy Phrase sessions recorded in January of 2024 (assuming the BciPy folder naming convention).
  - Glob patterns can also include nested directories (ex. `"*/*Matrix_Copy*"`).
- `p` : path to the parameters.json file used to run the simulation. These parameters will be applied to
  all raw_data files when loading.
- `sm`: the signal model argument is necessary to generate evidence

### Architecture

![simDiagram.png](res/simDiagram.png)


