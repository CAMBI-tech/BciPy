"""Demonstrates converting BciPy data output to other EEG formats.

To use at bcipy root,

    `python bcipy/io/demo/demo_convert.py -d "path://to/bcipy/data/folder"`
"""
import os
from bcipy.io.convert import convert_to_bids, ConvertFormat
from bcipy.io.load import load_experimental_data

EXCLUDED_TASKS = ['Report', 'Offline', 'InterTask']


def convert_study_to_bids(directory: str, experiment_id: str, format: ConvertFormat = ConvertFormat.BV):
    """Converts the data in the study folder to BIDS format."""
    participant_counter = 1
    for participant in os.listdir(directory):
        for experiment in os.listdir(os.path.join(directory, participant)):
            if experiment == experiment_id:
                # loop over the experiment sessions
                session_counter = 1
                for session in os.listdir(os.path.join(directory, participant, experiment)):
                    # convert the session data to BIDS format
                    for run in os.listdir(os.path.join(directory, participant, experiment, session)):
                        if run not in EXCLUDED_TASKS:
                            convert_to_bids(
                                data_dir=os.path.join(directory, participant, experiment, session, run),
                                participant_id=f'0{participant_counter}',
                                session_id=f'0{session_counter}',
                                run_id=run,
                                output_dir=f'./bids/{experiment}/',
                                format=format.value
                            )
                    session_counter += 1
        participant_counter += 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--directory',
        help='Path to the directory with raw_data to be converted',
        required=False)

    args = parser.parse_args()

    path = args.directory
    if not path:
        path = load_experimental_data()

    convert_to_bids(
        data_dir=path,
        participant_id='01',
        session_id='01',
        run_id='01',
        output_dir='./bids/',
        format=ConvertFormat.BV
    )
