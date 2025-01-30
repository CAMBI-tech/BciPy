## RSVP Simulator

### Overview

This Simulator module aims to automate experimentation by sampling EEG data from prior sessions and running given models in a task loop, thus simulating a live session.

### Run steps

`main.py` is the entry point for program. After following `BciPy` readme steps for setup, run the module from terminal:

```
(venv) $ bcipy-sim -h
usage: bcipy-sim [-h] [-i] [--gui] [-d DATA_FOLDER] [-m MODEL_PATH] [-p PARAMETERS] [-n N] [-s SAMPLER] [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -i, --interactive     Use interactive command line for selecting simulator inputs
  --gui                 Use interactive GUI for selecting simulator inputs
  -d DATA_FOLDER, --data_folder DATA_FOLDER
                        Raw data folders to be processed. Multiple values can be provided, or a single parent folder.
  -m MODEL_PATH, --model_path MODEL_PATH
                        Signal models to be used. Multiple models can be provided.
  -p PARAMETERS, --parameters PARAMETERS
                        Parameter File to be used
  -n N                  Number of times to run the simulation
  -s SAMPLER, --sampler SAMPLER
                        Sampling strategy
  --sampler_args SAMPLER_ARGS
                        Sampler args structured as a JSON string.
  -o OUTPUT, --output OUTPUT
                        Sim output path
```

For example,
`$ python bcipy/simulator -d my_data_folder/ -p my_parameters.json -m my_models/ -n 5`

#### Program Args

- `i` : Interactive command line interface. Provide this flag by itself to be prompted for each parameter.
- `gui`: A graphical user interface for configuring a simulation. This mode will output the command line arguments which can be used to repeat the simulation.
- `d` : Raw data folders to be processed. One ore more values can be provided. Each session data folder should contain
  _raw_data.csv_, _triggers.txt_, _parameters.json_. These files will be used to construct a data pool from which simulator will sample EEG and other device responses. The parameters file in each data folder will be used to check compatibility with the simulation/model parameters.
- `p` : path to the parameters.json file used to run the simulation. These parameters will be applied to
  all raw_data files when loading. This file can specify various aspects of the simulation, including the language model to be used, the text to be spelled, etc. Timing-related parameters should generally match the parameters file used for training the signal model(s).
- `m`: Path to a pickled (.pkl) signal model. One or more models can be provided.
- `n`: Number of simulation runs
- `o`: Output directory for all simulation artifacts.
- `s`: Sampling strategy to use; by default the TargetNonTargetSampler is used. The value provided should be the class name of a Sampler.
- `sampler_args`: Arguments to pass in to the selected Sampler. Some samplers can be customized with further parameters. These should be structured as a JSON dictionary mapping keys to values. For example: `--sampler_args='{"inquiry_end": 4}'`

#### Sim Output Details

Output folders are generally located in the `data/simulator` directory, but can be configured per simulation. Each simulation will create a new directory. The directory name will be  prefixed with `SIM` and will include the current date and time.

- `parameters.json` captures params used for the simulation.
- `sim.log` is a log file for the simulation; metrics will be output here.
- `summary_data.json` summarizes session data from each of the runs into a single data structure.
- `metrics.png` boxplots for several metrics summarizing all simulation runs.

A directory is created for each simulation run. The directory contents are similar to the session output in a normal bcipy task. Each run directory contains:

- `run_{n}.log` log file specific to the run, where n is the run number.
- `session.json` session data output for the task, including evidence generated for each inquiry and overall metrics.
- `session.xlsx` session data summarized in an excel spreadsheet with charts for easier visualization.

## Main Components

* Task - a simulation task to be run (ex. RSVP Copy Phrase)
* TaskRunner - runs one or more iterations of a simulation
* TaskFactory - constructs the hierarchy of objects needed for the simulation.
* DataEngine - loads data to be used in a simulation and provides an API to query for data.
* DataProcessor - used by the DataEngine to pre-process data. Pre-processed data can be classified by a signal model.
* Sampler - strategy for sampling data from the data pool stored in the DataEngine.

## Device Support

The simulator is structured to support evidence from multiple devices (multimodal). However, it currently only includes processing for EEG device data. To provide support for models trained on data from other devices (ex. Gaze), a `RawDataProcessor` must be added for that device. The Processor pre-processes data collected from that device and prepares it for sampling. A `RawDataProcessor` is matched up to a given signal model using that model's metadata (metadata.device_spec.content_type). See the `data_process` module for more details.

## Parameters

The parameters file is used to configure various aspects of the of the simulation. Timing-related parameters should generally match the parameters file used for training the signal model(s). Following are some specific parameters that you may want to modify, depending on the goals of a particular simulation:

* `task_text` - the text to spell.
* `lang_model_type` - language model to use in the simulation.
* `summarize_session` - if set to true a session.xlsx summary will be generated for each simulation run.

### Stoppage Criteria

Parameters which define task stoppage criteria are important to ensure that the simulation runs to completion without getting stuck in an infinite loop. The values for these parameters may also affect analysis of results.

* `min_inq_len` - Specifies the minimum number of inquiries to present before making a decision in copy/spelling tasks.
* `max_inq_len` - maximum number of inquiries to display before stopping the task.
* `max_selections` - The maximum number of selections for copy/spelling tasks. The task will end if this number is reached.
* `max_incorrect` - The maximum number of consecutive incorrect selections for copy/spelling tasks. The task will end if this number is reached.
* `max_inq_per_series` - Specifies the maximum number of inquiries to present before making a decision in copy/spelling tasks

## GUI

A simulation can be started using a graphical user interface.

`$ bcipy-sim --gui`

This provides a way to explore the file system when providing the parameters.json file, simulation model, and input data sources. After all required inputs have been provided the user can initiate the simulation from the GUI interface. The command line used to run the simulation are output to the console prior to the run making it easier to start subsequent simulations with the same set of arguments.

## Current Limitations

* Only provides EEG support
* Only one sampler maybe provided for all devices. Ideally we should support a different sampling strategy for each device.
* Only Copy Phrase is currently supported.