"""Update triggers for EEGlab

How to use!

1. Ensure you have BciPy installed
2. pip install click
3. Then you can use the script!
    a. python update_for_eeglab.py 
    b. python update_for_eeglab.py --help
    c. python update_for_eeglab.py --directory <path to relevant BciPy sessions. Must contain 1 or more session files with triggers.txt in them>
    d. python update_for_eeglab.py --static <static offset to add to all triggers. Default is 0.1>
"""

import click
from bcipy.helpers.load import fast_scandir


DEFAULT_STATIC = 0.1
UPDATED_TRIGGER_FILENAME = 'triggers_updated.txt'
TRIGGER_FILENAME = 'triggers.txt'

@click.command()
@click.option('--directory', prompt='Provide a path to a collection of session folders to export trigger file for import into EEGLab',
              help='The path to data')
@click.option('--static', default=DEFAULT_STATIC, help='The static value to add to the trigger values. Default is 0.1')
def update_triggers(directory, static):
    """Update Triggers."""
    print(f'Using static offset of {static}')
    sessions = fast_scandir(directory)
    for session in sessions:
        try:
            trigger_file = f'{session}/{TRIGGER_FILENAME}'
            triggers = read_triggers(trigger_file, static)
            trigger_file_updated = f'{session}/{UPDATED_TRIGGER_FILENAME}'
            write_triggers(triggers, path=trigger_file_updated)
        except Exception as e:
            print(f'Convert Failure for {session}. Skipping. Ensure the directory provided contains session folders'
                  ' with triggers.txt in them. Not individual session files like when training a model.\n')


def read_triggers(triggers_file, static):
    """Read in the triggers.txt file. Convert the timestamps to be in
    aqcuisition clock units using the offset listed in the file (last entry).
    Returns:
    --------
        list of (symbol, targetness, stamp) tuples."""

    with open(triggers_file) as trgfile:
        records = [line.split(' ') for line in trgfile.readlines()]
        (_cname, _ctype, cstamp) = records[0]
        records.pop(0)
        # (_acq_name, _acq_type, acq_stamp) = records[-1]
        static_offset = static
        offset = float(cstamp) + static_offset

        corrected = []
        for i, (name, trg_type, stamp) in enumerate(records):
            if i < len(records) - 1:
                # omit offset record for plotting
                corrected.append((name, trg_type, float(stamp) + offset))
        return corrected

def write_triggers(triggers, path):
    with open(path, 'w+', encoding='utf-8') as trigger_write:
        for trigger in triggers:
            trigger_write.write(f'{trigger[0]} {trigger[1]} {trigger[2]}\n')
    print(f'Updated triggers written to {path}')


if __name__ == '__main__':
    update_triggers()