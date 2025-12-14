import argparse
from pathlib import Path

from bcipy.config import DEFAULT_PARAMETERS_FILENAME
from bcipy.io.load import load_json_parameters
from bcipy.simulator.util.switch_utils import (generate_raw_data,
                                               simulate_raw_data)


def main() -> Path:
    """Main method used to generate a raw data file for a switch device."""
    parser = argparse.ArgumentParser(
        description="Create raw marker data for a given session.")
    parser.add_argument("data_folder",
                        type=Path,
                        help="Data directory (must contain triggers.txt file)")
    mock_help = "Mock data (use when button presses are not recorded in trigger file)"
    parser.add_argument("-m", "--mock", action='store_true', help=mock_help)
    parser.add_argument(
        "-p",
        "--parameters",
        type=Path,
        required=False,
        help="Optional parameters file to use when mocking data.")
    args = parser.parse_args()
    data_dir = args.data_folder

    if args.mock:
        params_file = args.parameters or Path(data_dir,
                                              DEFAULT_PARAMETERS_FILENAME)
        params = load_json_parameters(str(params_file), value_cast=True)
        return simulate_raw_data(data_dir, params)
    else:
        return generate_raw_data(data_dir)


if __name__ == '__main__':
    print(str(main()))
