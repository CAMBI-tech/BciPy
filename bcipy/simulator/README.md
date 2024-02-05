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

``python3 bcipy/simulator -d "pathToDataFolder1" -d "pathToOptionalDataFolder2" -sm "pathToSignalModel.pkl"``

Then view generated LOG file for analysis

#### Program Args

- `d` : the data folder argument is necessary. This data will be used to construct data pool from
  which simulator will
  sample EEG responses. Should contain
  _raw_data.csv_, _triggers.txt_, _parameters.json_
- `sm`: the signal model argument is necessary to generate evidence

### Architecture

![simDiagram.png](res/simDiagram.png)


