# BCI TASKS
-----------

These are the experimental tasks that can be implemented.

## Modes
---------
Within `tasks/` there are folders for each of the supported modes, and within them, the supported experiment types. To add new modes, create a folder for it and place the tasks in files within it. Be sure to add it to the `start_task` file at the root to be able execute it! An entry must also be added to the task_registry TaskType
enum. This updates the GUI (BCInterface.py) as well makes the task available to `start_task`.

Currently, these are the modes and experiment types implemented:

*RSVP* 

> Calibration
> Copy Phrase
> Alert Tone
> Inter-Inquiry Feedback Calibration


## Start Task
-------------

*Start Task* 

Start Task takes in Display [object], parameters [dict], file save [str-path] and task type [dict]. Using the
task type, start_task() will route to the correct paradigm (RSVP, SSVEP, MATRIX) and mode (Calibration, Copy Phrase, etc.)

It is called in the following way:


```
	from bcipy.task import start_task

    start_task(
       	display_window,
        data_acquisition_client,
        parameters,
        file_save)

```

It will throw an error that the task isn't implemented.
