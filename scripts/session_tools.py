"""Tools for viewing and debugging session.json data"""

import contextlib
# prevents pygame from outputting to the console on import.
with contextlib.redirect_stdout(None):
    import pygame
import json
import os
from collections import Counter

from bcipy.helpers.load import load_json_parameters
from bcipy.helpers.task import alphabet


def session_data(data_dir: str, alp=None):
    """Returns a dict of session data transformed to map the alphabet letter
    to the likelihood when presenting the evidence. Also removes attributes
    not useful for debugging."""

    # Get the alphabet based on the provided parameters (txt or icon).
    parameters = load_json_parameters(
        os.path.join(data_dir, "parameters.json"), value_cast=True)
    if parameters.get('is_txt_sti', False):
        parameters['is_txt_stim'] = parameters['is_txt_sti']
    
    if not alp:
        alp = alphabet(parameters=parameters)

    session_path = os.path.join(data_dir, "session.json")
    with open(session_path, 'r') as json_file:
        data = json.load(json_file)
        data['copy_phrase'] = parameters['task_text']
        for epoch in data['epochs'].keys():
            for trial in data['epochs'][epoch].keys():
                likelihood = dict(
                    zip(alp, data['epochs'][epoch][trial]['likelihood']))

                # Remove unused properties
                unused = ['eeg_len', 'timing_sti', 'triggers', 'target_info', 'copy_phrase']
                removeProps(data['epochs'][epoch][trial], unused)
                
                data['epochs'][epoch][trial]['stimuli'] = data['epochs'][
                    epoch][trial]['stimuli'][0]

                # Associate letters to values
                data['epochs'][epoch][trial]['lm_evidence'] = dict(
                    zip(alp, data['epochs'][epoch][trial]['lm_evidence']))
                data['epochs'][epoch][trial]['eeg_evidence'] = dict(
                    zip(alp, data['epochs'][epoch][trial]['eeg_evidence']))
                data['epochs'][epoch][trial]['likelihood'] = likelihood

                # Display the 5 most likely values.
                data['epochs'][epoch][trial]['most_likely'] = dict(
                    Counter(likelihood).most_common(5))

        return data

def removeProps(data, proplist):
    for prop in proplist:
        if prop in data:
            data.pop(prop)

def main(data_dir: str, alphabet: str):
    """Transforms the session.json file in the given directory and prints the
    resulting json."""
    print(json.dumps(session_data(data_dir, alphabet), indent=4))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Opens session.json file for analysis.")

    parser.add_argument(
        '-p', '--path', help='path to the data directory', default=None)
    parser.add_argument(
        '-a', '--alphabet', help='alphabet (comma-delimited string of items)', default=None)

    args = parser.parse_args()
    path = args.path
    if not path:
        from tkinter import Tk
        from tkinter import filedialog

        root = Tk()
        root.withdraw()
        path = filedialog.askdirectory(
            parent=root, initialdir="/", title='Please select a directory')

    alp = None
    if args.alphabet:
        alp = args.alphabet.split(",")
    main(path, alp)
