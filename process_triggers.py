import click

from pathlib import Path
from bcipy.helpers.convert import convert_to_edf
from bcipy.helpers.triggers import TriggerHandler, Trigger, TriggerType, FlushFrequency, trigger_decoder
from bcipy.helpers.load import (
    load_json_parameters,
)

@click.command()
@click.option('--directory', prompt='Provide a path to a data folders to be updated for export to EEGLab',
              help='The path to data folder containing a triggers.txt file')
def process_data(directory):
    parameters_filename = f'{directory}/parameters.json'
    parameters = load_json_parameters(parameters_filename, value_cast=True)
    static_offset = parameters.get("static_trigger_offset")

    trigger_type, trigger_timing, trigger_label = trigger_decoder(
        str(Path(directory, f'triggers.txt')), remove_pre_fixation=False, offset=static_offset, exclusion=[TriggerType.OFFSET])

    convert_to_edf(directory, overwrite=True, write_targetness=True)
    triggers = [Trigger(label, TriggerType(triggertype), time) for label, triggertype, time in zip(trigger_label, trigger_type, trigger_timing)]
    trigger_handler = TriggerHandler(directory, 'updated_triggers.txt', FlushFrequency.END)
    trigger_handler.add_triggers(triggers)
    trigger_handler.close()

if __name__ == '__main__':
    process_data()
