# BCI TASKS
-----------

These are the tasks that can be run to collect experimental data.

## Paradigms
------------
Within `task/` there are folders for each of the supported paradigms, and within them, the supported modes. To add new paradigms, create a folder for it and place the tasks in files within it. Be sure to add it to the `start_task` file at the root to be able execute it! An entry must also be added to the task_registry TaskType
enum. This updates the GUI (BCInterface.py) and makes the task available to `start_task`.

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



## Start Task
-------------

Start Task takes in Display [object], parameters [dict], file save [str-path] and task type [dict]. Using the
task type, start_task() will route to the correct paradigm (RSVP, SSVEP, MATRIX) and mode (Calibration, Copy Phrase, etc.)

It is called in the following way:


```
	from bcipy.task.start_task import start_task

    start_task(
       	display_window,
        data_acquisition_client,
        parameters,
        file_save)

```

It will throw an error if the task isn't implemented.
