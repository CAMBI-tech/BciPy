import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from bcipy.config import SESSION_DATA_FILENAME
from bcipy.helpers.session import read_session


@dataclass
class SimState:
    """ Represents the state of a current session during simulation """
    target_symbol: str
    current_sentence: str
    target_sentence: str
    display_alphabet: List[str]
    inquiry_n: int
    series_n: int


def get_inquiry(session_dir: str, n: int) -> Dict[str, Any]:
    """Extracts an inquiry from a session.json file. Useful for debugging
    simulator output."""
    session = read_session(f"{session_dir}/{SESSION_DATA_FILENAME}")
    inq = session.all_inquiries[n]
    return inq.stim_evidence(session.symbol_set)


def main():
    """Command line program to get data from a single inquiry within a session."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--data_folder",
                        type=Path,
                        required=True,
                        help="Folder with the session.json file")
    parser.add_argument("-n",
                        type=int,
                        required=False,
                        default=1,
                        help="inquiry number")

    args = parser.parse_args()
    inq = get_inquiry(args.data_folder, args.n)
    print(json.dumps(inq, indent=2))


if __name__ == '__main__':
    # python bcipy/simulator/helpers/state.py -h
    main()
