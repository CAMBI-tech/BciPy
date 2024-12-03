"""Update parameters files to latest."""

import argparse
import json
import shutil
from pathlib import Path

from bcipy.helpers.parameters import DEFAULT_PARAMETERS_PATH, Parameters


def update(params_path: str) -> None:
    """Update the parameters at the given path"""
    default_params = Parameters(DEFAULT_PARAMETERS_PATH, cast_values=True)

    # open as json, not Parameters, to avoid validation checks
    with open(params_path, 'r', encoding='utf8') as json_file:
        params = json.load(json_file)

    # rename attributes if needed
    mapping = {"readableName": "name", "recommended_values": "recommended"}

    updated_params = {}
    for key, entry in params.items():
        editable = False
        if key in default_params:
            editable = default_params.get_entry(key)['editable']
        value = {mapping.get(k, k): v for k, v in entry.items()}

        if "editable" not in value:
            value["editable"] = editable
        updated_params[key] = value

    # overwrite json file if needed
    original_path = Path(params_path)
    shutil.copyfile(original_path,
                    Path(original_path.parent, "parameters_original.json"))

    with open(params_path, 'w', encoding='utf8') as json_path:
        json.dump(updated_params, json_path, ensure_ascii=False, indent=2)

    # load from overwritten file
    parameters = Parameters(source=params_path, cast_values=True)

    added_params = [
        key for key, change in default_params.diff(parameters).items()
        if change.original_value is None
    ]

    if added_params:
        print(
            f"Adding missing parameters using default values: {added_params}")
        parameters.add_missing_items(default_params)

    parameters.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",
                        "--parameters",
                        type=Path,
                        required=False,
                        help="Parameter File to be used")
    args = parser.parse_args()
    update(args.parameters)
