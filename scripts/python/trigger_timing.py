import os
STATIC = 0.1

def main(path: str):
    """Run the viewer gui

    Parameters:
    -----------
        data_file - raw_data.csv file to stream.
        seconds - how many seconds worth of data to display.
        downsample_factor - how much the data is downsampled. A factor of 1
            displays the raw data.
    """
    # data_file = os.path.join(path, 'raw_data.csv')
    trg_file = os.path.join(path, 'test.txt')
    # data, device_info = file_data(data_file)
    triggers = read_triggers(trg_file)
    return triggers


def read_triggers(triggers_file):
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
        static_offset = STATIC
        offset = float(cstamp) + static_offset

        corrected = []
        for i, (name, trg_type, stamp) in enumerate(records):
            if i < len(records) - 1:
                # omit offset record for plotting
                corrected.append((name, trg_type, float(stamp) + offset))
        return corrected


def decompose_timing(triggers):

    inquires = {}
    inquiry = 0
    start = False
    for trigger in triggers:
        if trigger[1] == 'fixation':
            inquiry += 1
            start = True
            inquires[inquiry] = []
        elif start and trigger[1] != 'fixation' and trigger[1] != 'preview' and trigger[1] != 'prompt':
            inquires[inquiry].append(float(trigger[2]))
    
    return inquires


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Graphs trigger data from a bcipy session to visualize system latency"
    )
    parser.add_argument(
        '-p', '--path', help='path to the data directory', default=None)
    args = parser.parse_args()
    path = args.path
    if not path:
        from tkinter import filedialog
        from tkinter import Tk
        root = Tk()
        path = filedialog.askdirectory(
            parent=root, initialdir="/", title='Please select a directory')

    response = main(path)
    timing = decompose_timing(response)

    rate = []
    for _, key in enumerate(timing):
        values = timing[key]
        previous_value = False
        for value in values:
            if previous_value:
                rate.append(float(value - previous_value))
                previous_value = value
            else:
                previous_value = value    

    print(f'Average: {sum(rate) / len(rate)} Min: {min(rate)} Max: {max(rate)}')
