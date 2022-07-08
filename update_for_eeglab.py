import click
from bcipy.helpers.load import fast_scandir
STATIC = 0.062

@click.command()
@click.option('--directory', prompt='Provide a path to a collection of session folders to export trigger file for import into EEGLab',
              help='The path to data')
def update_triggers(directory):
    """Update Triggers."""

    sessions = fast_scandir(directory)
    for session in sessions:
        try:
            trigger_file = f'{session}/triggers.txt'
            triggers = read_triggers(trigger_file)
            trigger_file_updated = f'{session}/updated_triggers.txt'
            write_triggers(triggers, path=trigger_file_updated)
        except Exception as e:
            print(f'Convert Failure for {session}. Skipping.')
            print(f'Error={e}')

    print('Complete! Updated triggers written to corresponding session folders.')


def read_triggers(triggers_file):
    """Read in the triggers.txt file. Convert the timestamps to be in
    acquisition clock units using the offset listed in the file (last entry).
    Returns:
    --------
        list of (symbol, targetness, stamp) tuples."""

    with open(triggers_file) as trgfile:
        records = [line.split(' ') for line in trgfile.readlines()]
        (_cname, _ctype, cstamp) = records[0]
        records.pop(0)
        static_offset = STATIC
        offset = float(cstamp) + static_offset

        corrected = []
        for i, (name, trg_type, stamp) in enumerate(records):
            if i < len(records) - 1:
                # omit offset record for plotting
                corrected.append((name, trg_type, float(stamp) + offset))
        return corrected

def write_triggers(triggers, path='triggers_update.txt'):
    with open(path, 'w+', encoding='utf-8') as trigger_write:
        for trigger in triggers:
            trigger_write.write(f'{trigger[0]} {trigger[1]} {trigger[2]}\n')


if __name__ == '__main__':
    update_triggers()