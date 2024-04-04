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

``python3 bcipy/simulator -d "pathToDataFolderWrapper" -p "pathToParametersFile" -sm "pathToSignalModel.pkl"``

Then view the appropriate output folder for LOG file(s) and metrics.

#### Program Args

- `d` : the data wrapper folder argument is necessary. This folder is expected to contain 1 or more session folders. Each session folder should contain
  _raw_data.csv_, _triggers.txt_, _parameters.json_. These files will be used to construct data pool from which simulator will sample EEG responses. 
- `g` : optional glob filter that can be used to select a subset of data.
  - Ex. `"*Matrix_Copy*Jan_2024*"` will select all data for all Matrix Copy Phrase sessions recorded in January of 2024 (assuming the BciPy folder naming convention).
  - Glob patterns can also include nested directories (ex. `"*/*Matrix_Copy*"`).
- `p` : path to the parameters.json file used to run the simulation. These parameters will be applied to
  all raw_data files when loading.
- `sm`: the signal model argument is necessary to generate evidence

#### Sim Output Details

Output folders are generally located at [simulator/generated](). 

- `result.json` captures metrics from the simulation and all the state from the simulation (like _session.json_)
- `parameters.json` captures params used, a combination of parameters inputted by you and the `sim_parameters.json`
- Single run Folder name: _SIM_month-day-hour:min_hash_
- Multi Run Folder name: _SIM_MULTI_runCount_month-day-hour:min_


#### Configuring Parameters

There are various parameters that you can tweak in `sim_parameters.json` to run simulator multiple times with the same args, specify lang_model type ...


### Architecture

![simDiagram.png](res/simDiagram.png)


