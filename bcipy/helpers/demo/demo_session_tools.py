"""Tools for viewing and debugging session.json data"""
# pylint: disable=invalid-name
import json
from pathlib import Path
from bcipy.helpers.session import session_data, session_db, session_csv, session_excel
from bcipy.gui.file_dialog import ask_directory


def main(data_dir: str):
    """Transforms the session.json file in the given directory and prints the
    resulting json."""
    print(json.dumps(session_data(data_dir), indent=4))


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Opens session.json file for analysis. "
        "Optionally creates a sqlite database summarizing the data."
    )

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
        db_name = str(Path(path, "session.db"))
        session_db(path, db_name=db_name)
        if args.csv:
            session_csv(db_name=db_name, csv_name=str(Path(path, "session.csv")))
        if args.charts:
            session_excel(db_name=db_name,
                          excel_name=str(Path(path, "session.xlsx")))
        if not args.db:
            os.remove(db_name)
    else:
        main(path)
