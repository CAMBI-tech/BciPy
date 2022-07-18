import imp
import click
from bcipy.helpers.load import fast_scandir
from bcipy.helpers.convert import convert_to_edf


static_offset = 0.62


@click.command()
@click.option('--directory', prompt='Provide a path to a collection of data folders to be updated to BciPy 2.0',
              help='The path to data')
def update_triggers(directory):
    """Update Triggers."""

    particpants = fast_scandir(directory)
    for part_dir in particpants:
        eeg_level_dir = fast_scandir(part_dir)
        for top_level in eeg_level_dir:
            session_dirs = fast_scandir(top_level)
            for session in session_dirs:
                trigger_file = f'{session}\\triggers.txt'
                triggers = load_triggers(trigger_file)
                test = correct_triggers(triggers)
                write_triggers(test, path=trigger_file)
                edf_path = convert_to_edf(
                    session,
                    use_event_durations=False,
                    write_targetness=True,
                    overwrite=True,
                    annotation_channels=4)

def load_triggers(trigger_path):
    # Get every line of triggers.txt
    with open(trigger_path, 'r+') as text_file:
        trigger_txt = [line.split() for line in text_file]
    return trigger_txt

def correct_triggers(triggers):
    calibration_trigger = triggers.pop(0)
    offset_trigger = triggers.pop(-1)

    new_offset_value = float(offset_trigger[-1]) - float(calibration_trigger[-1])
    offset = ['starting_offset', 'offset', new_offset_value]

    new_triggers = [offset]
    for trigger in triggers:
        label = trigger[0]
        targetness = trigger[1]
        time = trigger[2]
        if targetness == 'first_pres_target':
            targetness = 'prompt'
        new_triggers.append([label, targetness, time])
    return new_triggers

def write_triggers(triggers, path='triggers_update.txt'):
    with open(path, 'w+', encoding='utf-8') as trigger_write:
        for trigger in triggers:
            trigger_write.write(f'{trigger[0]} {trigger[1]} {trigger[2]}\n')



if __name__ == '__main__':
    update_triggers()