"""Tools for viewing and debugging session.json data"""
# pylint: disable=invalid-name
import json
from pathlib import Path

from bcipy.config import SESSION_DATA_FILENAME, SESSION_SUMMARY_FILENAME
from bcipy.gui.file_dialog import ask_directory
from bcipy.data.session import (read_session, session_csv, session_data,
                                   session_db, session_excel)


def main(data_dir: str):
    """Transforms the session.json file in the given directory and prints the
    resulting json."""
    print(json.dumps(session_data(data_dir), indent=4))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Opens session.json file for analysis. "
        "Optionally creates a sqlite database summarizing the data.")

    parser.add_argument('-p',
                        '--path',
                        help='path to the data directory',
                        default=None)
    parser.add_argument('--db',
                        help='create sqlite database',
                        action='store_true')
    parser.add_argument('--csv',
                        help='create a csv file from the database',
                        action='store_true')
    parser.add_argument('--charts',
                        help='create an Excel spreadsheet with charts',
                        action='store_true')

    args = parser.parse_args()
    path = args.path
    if not path:
        path = ask_directory()

    if args.db or args.csv or args.charts:
        session = read_session(Path(path, SESSION_DATA_FILENAME))
        if args.db:
            session_db(session, db_file=str(Path(path, "session.db")))
        if args.csv:
            session_csv(session, csv_file=str(Path(path, "session.csv")))
        if args.charts:
            session_excel(session, excel_file=str(Path(path, SESSION_SUMMARY_FILENAME)))
    else:
        main(path)
