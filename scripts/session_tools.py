"""Tools for viewing and debugging session.json data"""
# pylint: disable=invalid-name
import json
from bcipy.helpers.session import session_data, session_db


def main(data_dir: str, alphabet: str):
    """Transforms the session.json file in the given directory and prints the
    resulting json."""
    print(json.dumps(session_data(data_dir, alphabet), indent=4))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Opens session.json file for analysis. Optionally creates a sqlite database summarizing the data.")

    parser.add_argument('-p',
                        '--path',
                        help='path to the data directory',
                        default=None)
    parser.add_argument('--db', help='create sqlite database', default=False)
    parser.add_argument('-a',
                        '--alphabet',
                        help='alphabet (comma-delimited string of items)',
                        default=None)

    args = parser.parse_args()
    path = args.path
    if not path:
        from tkinter import Tk
        from tkinter import filedialog

        root = Tk()
        root.withdraw()
        path = filedialog.askdirectory(parent=root,
                                       initialdir="/",
                                       title='Please select a directory')

    alp = None
    if args.alphabet:
        alp = args.alphabet.split(",")

    if args.db:
        session_db(path, alp=alp)
    else:
        main(path, alp)
