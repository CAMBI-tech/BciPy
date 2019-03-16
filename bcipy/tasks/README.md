# BCI TASKS
-----------

These are the experimental tasks that can be implemented.

## Modes
---------
Within `tasks/` there are folders for each of the supported modes, and within them, the supported experiment types. To add new modes, create a folder for it and place the tasks in files within it. Be sure to add it to the `start_task` file at the root to be able execute it! An entry must also be added to the task_registry ExperimentType
enum. This updates the GUI (RSVPKeyboard.py) as well makes the task available to `start_task`.

Currently, these are the modes and experiment types implemented:

*RSVP* 

> Calibration
> Copy Phrase
> Icon to Icon
> Icon to Word
> Alert Tone
> Intersequence Feedback Calibration


## Start Task
-------------

*Start Task* 

Start Task takes in Display [object], parameters [dict], file save [str-path] and task type [dict]. Using the
task type, start_task() will route to the correct mode (RSVP, SSVEP, MATRIX) and experiment type (Calibration, Copy Phrase, etc.)

It is called in the following way:

```
	from bci_tasks.start_task import start_task

    start_task(
       	display_window,
        task_type,
        parameters,
        file_save)

```

It will throw an error that the task isn't implemented if given a mode or experiment type that's unavailable to it. 


## Using Images
---------------

1. You make a folder in the `static/images` dir with your .pngs
2. Point to that folder in the parameters.json file
3. Set is_txt_sti to false