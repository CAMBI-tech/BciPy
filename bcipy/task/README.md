# BCI TASKS
-----------

These are the tasks that can be run to collect experimental data.

## Paradigms
------------
Within `task/` there are folders for each of the supported paradigms, and within them, the supported modes. To add new paradigms, create a folder for it and place the tasks in files within it. An entry must also be added to the task_registry TaskType and/or TaskMode enum. This updates the GUI (BCInterface.py) and makes the task available to the BciPy Client.

Currently, these are the supported paradigms and modes:

### *Paradigm: RSVP* 

##### Mode: Calibration

> Calibration: Used to calibrate the RSVP paradigm for a user
> Time Test Calibration: Used to verify the timing of the RSVP paradigm

##### Mode: Copy Phrase

> Copy Phrase: Used to copy a phrase using the RSVP paradigm (e.g. P300 Speller) on data from a P300 calibration

### *Paradigm: Matrix* 

##### Mode: Calibration

> Calibration: Used to calibrate the Matrix paradigm for a user
> Time Test Calibration: Used to verify the timing of the Matrix paradigm

##### Mode: Copy Phrase

> Copy Phrase: Used to copy a phrase using the Matrix paradigm (e.g. P300 Speller) on data from a P300 calibration


### *Paradigm: VEP*

##### Mode: Calibration

> Calibration: Used to calibrate the VEP paradigm for a user. Note this has not been extensively tested, use with caution.


## Actions
-----------

### BciPy Calibration Report Action

This action is used to generate a calibration report for a user. It will generate a report with the following information:

- Session Summary
  - Task / Paradigm (RSVP, Matrix, or VEP)
  - AUC (if available)
  - Amplifier
  - Data Location
- Signal Report
  - Signal Artifact Report
    - Determine the percentage of data that is artifact based on the artifact thresholds
      - See `bcipy.signal.evaluate.artifact.py` for more information, to visualize the artifacts or to adjust the thresholds.
  - Signal Plots (target / non-target)
    - ERP
    - Topographic Maps

The report will be saved in the protocol directory. A demo of the action can be used offline by running the following command:

```bash
python bcipy/task/demo/demo_calibration_report.py
```

This will prompt the user to select a protocol directory to generate a calibration report for. The report will be saved in the protocol directory.

### Offline Analysis Action

This action is used to run an offline analysis on the data collected during a session. It will run the command line version of the BciPy client on the data collected during the session. The results will be saved in the session directory.

## Running Tasks using the SessionOrchestrator

The `SessionOrchestrator` is a class that can be used to run a Protocol (sequence of Tasks/Actions). The core BciPy client and GUI use this class and resulting data strucutures. It will run the tasks in the order defined, handle the transition between tasks, and persist data. There are several optional arguments that can be provided to the orchestrator:

experiment_id: str
    This is used to load any defined protocols or field collections. If no experiment_id is provided, a default will be used, and the orchestrator will run any tasks in the order they were added.
user: str
    The user ID to associate with the session data. By default, this is DEFAULT_USER.
parameters_path: str
    The path to the BciPy parameters file. By default, this is DEFAULT_PARAMETERS_PATH, located at bcipy/parameters/parameters.json. 
parameters: Parameters
    A Parameters object to use for the Tasks. If provided, this will override the parameters_path.
fake: bool
    If True, the Tasks will run in fake mode. This is useful for testing or debugging paradigms with streaming data or models. By default, this is False.
alert: bool
    If True, after a Task execution the orchestrator will alert experimenters when a task is complete. By default, this is False.
visualize: bool
    If True, after a Task execution the orchestrator will visualize data. By default, this is False. This only works for EEG data with target/non-target labels.


The data will be saved in the following format in the specified data_save_loc (default is bcipy/data/):

```
<!-- The following would be a single orchestration run with two Tasks  -->
data_save_loc/
    user_id/
        date/
          experiment_id/
              run_id <datetimestamp>/
                  task_id/
                      logs/
                          task_log_data
                      task_data (e.g. acquisition data, parameters, visualizations)
                  task_id/
                      logs/
                          task_log_data
                      task_data (e.g. acquisition data, parameters, visualizations)
                  logs/
                      protocol_log_data
                  protocol_data (system data, protocol/tasks executed)
```

### Usage manually

```python
from bcipy.task import Task
from bcipy.task import SessionOrchestrator
from bcipy.task.action import OfflineAnalysisAction
from bcipy.task.task_registry import TaskType
from bcipy.task.paradigm.rsvp.calibration import RSVPCalibration
from bcipy.task.paradigm.rsvp.copy_phrase import RSVPCopyPhrase

# Create a list of tasks to run, These should not be initialized.
tasks = [
    RSVPCalibration,
    OfflineAnalysisAction,
    RSVPCopyPhrase
]

# Create a SessionOrchestrator
orchestrator = SessionOrchestrator()

# add the tasks to the orchestrator
orchestrator.add_tasks(tasks)

# run the tasks
orchestrator.execute()
```

### Usage with a Protocol

```python
from bcipy.task import Task
from bcipy.task import SessionOrchestrator
from bcipy.task.action import OfflineAnalysisAction
from bcipy.task.task_registry import TaskType
from bcipy.task.paradigm.rsvp.calibration import RSVPCalibration
from bcipy.task.paradigm.rsvp.copy_phrase import RSVPCopyPhrase
from bcipy.task.protocol import parse_protocol

# Create a protocol. This would be extracted from a registered experiment.
example_protocol = 'RSVPCalibration -> OfflineAnalysisAction -> RSVPCopyPhrase'
# Parse the protocol into a list of tasks. This will raise an error if the TaskType is not registered.
tasks = parse_protocol(example_protocol)

# Create a SessionOrchestrator
orchestrator = SessionOrchestrator()

# add the tasks to the orchestrator
orchestrator.add_tasks(tasks)

# run the tasks
orchestrator.execute()
```

### Usage from experiment loading

Note: A new experiment must be registered in the `bcipy/parameters/experiments.json` file. The BCInterface may also be used to create a new named experiment.

```python
from bcipy.task import Task
from bcipy.task import SessionOrchestrator
from bcipy.task.action import OfflineAnalysisAction
from bcipy.task.task_registry import TaskType
from bcipy.task.paradigm.rsvp.calibration import RSVPCalibration
from bcipy.task.paradigm.rsvp.copy_phrase import RSVPCopyPhrase
from bcipy.task.protocol import parse_protocol
from bcipy.helpers.load import load_experiment

# Load an experiment from the registered experiments
experiment_name = 'default'
experiment = load_experiment(experiment_name)
# grab the protocol from the experiment and parse it
protocol = experiment['protocol']
tasks = parse_protocol(protocol)

# Create a SessionOrchestrator
orchestrator = SessionOrchestrator()

# add the tasks to the orchestrator
orchestrator.add_tasks(tasks)

# run the tasks
orchestrator.execute()
```

### Using orchestration to type using multiple copy phrases with different text and spelled letters

There are experiments in which multiple copy phrases would be used to test the performance of a user and the system over a variety of phrases. This is especially useful for testing the language model performance over different contexts and starting data. Additionally, this can be used to test the performance of the system over different spelling lengths and complexity.

If the `copy_phrase_location` parameter is set in the parameters.json file, the orchestrator will use the provided file to load the phrases to be copied in Tasks with the mode TaskMode.COPYPHRASE. The file should be a JSON file with the following format:

```json
{
    "Phrases": [
        ["This is the first phrase", 1],
        ["This is the second phrase", 2],
        ["This is the third phrase", 3]
    ]
}
```

Each phrase should be a list with the phrase as the first element (string) and the spelled letter count as the second element (integer). The orchestrator will iterate through the phrases in order, copying each one the specified number of times. If any phrases are remaining, the orchestrator will save the phrase list to the run directory for future use.








