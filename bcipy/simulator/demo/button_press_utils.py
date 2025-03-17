"""Utilities for working with button press data."""

import random
from pathlib import Path
from typing import List, Tuple

from bcipy import config
from bcipy.acquisition.datastream.mock.switch import switch_device
from bcipy.core.parameters import Parameters
from bcipy.core.raw_data import RawDataWriter
from bcipy.core.triggers import Trigger, TriggerType, trigger_decoder
from bcipy.display.main import ButtonPressMode, PreviewParams
from bcipy.helpers.acquisition import raw_data_filename


def pairwise(iterable):
    """
    pairwise('ABCDEFG') → AB BC CD DE EF FG
    https://docs.python.org/3/library/itertools.html#itertools.pairwise
    """
    iterator = iter(iterable)
    a = next(iterator, None)
    for b in iterator:
        yield a, b
        a = b


def partition_triggers(trigger_path: Path) -> List[List[Trigger]]:
    """Partition the triggers into inquiries.

    Returns
    -------
        list of entries where each entry represents the triggers for an inquiry.
    """
    trg_types, trg_times, trg_labels = trigger_decoder(
        str(trigger_path),
        remove_pre_fixation=False,
        apply_starting_offset=False)
    triggers = [
        Trigger(label=trg_labels[i],
                type=TriggerType(trg_types[i]),
                time=trg_times[i]) for i in range(len(trg_types))
    ]
    # index for each prompt trigger
    inq_start_indices = [
        i for i, trg in enumerate(triggers) if trg.type == TriggerType.FIXATION
    ]
    inquiry_triggers = []
    for i, j in pairwise(inq_start_indices):
        inquiry_triggers.append(triggers[i:j])
    inquiry_triggers.append(triggers[inq_start_indices[-1]:])

    return inquiry_triggers


def has_target(inquiry_triggers: List[Trigger]) -> bool:
    """Check if the list of inquiries contains a target."""
    for trg in inquiry_triggers:
        if trg.type == TriggerType.TARGET:
            return True
    return False


def time_range(inquiry_triggers: List[Trigger],
               time_flash: float) -> Tuple[float, float]:
    """Given a list of triggers for a given inquiry, determine the start and
    end timestamps of that inquiry."""
    return (inquiry_triggers[0].time, inquiry_triggers[-1].time + time_flash)


def should_press_button(inquiry_triggers: List[Trigger],
                        button_press_mode: ButtonPressMode) -> bool:
    """Determine if a marker should be written for the given inquiry
    depending on the presence of a target and the button press mode."""
    return (button_press_mode == ButtonPressMode.ACCEPT
            and has_target(inquiry_triggers)) or (
                button_press_mode == ButtonPressMode.REJECT
                and not has_target(inquiry_triggers))


def timestamp_within_inquiry(inquiry_triggers: List[Trigger],
                             parameters: Parameters) -> float:
    """Timestamp is a random value between inquiry start and end."""
    inq_start, inq_end = time_range(inquiry_triggers, parameters['time_flash'])
    return random.uniform(inq_start, inq_end)


def simulate_raw_data(data_dir: Path, parameters: Parameters):
    """Simulate what the raw_data file for a switch would generate.

    Reads through trigger data. For inquiries with target, outputs one or
    more markers within the inquiry timestamp range. Or if press to reject,
    output markers for inquiries without target.
    """

    spec = switch_device()
    raw_data_path = Path(data_dir, raw_data_filename(spec))

    if raw_data_path.exists():
        return raw_data_path

    # determine button mode from parameters
    button_press_mode = parameters.instantiate(PreviewParams).button_press_mode

    columns = ['timestamp', 'Marker', 'lsl_timestamp']
    rownum = 0
    with RawDataWriter(raw_data_path,
                       daq_type=spec.content_type,
                       sample_rate=spec.sample_rate,
                       columns=columns) as writer:
        for inquiry_triggers in partition_triggers(
                Path(data_dir, config.TRIGGER_FILENAME)):
            if should_press_button(inquiry_triggers, button_press_mode):
                rownum += 1
                stamp = timestamp_within_inquiry(inquiry_triggers, parameters)

                writer.writerow([rownum, 1.0, stamp])

    return raw_data_path
