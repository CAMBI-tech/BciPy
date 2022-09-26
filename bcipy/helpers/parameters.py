"""Module for functionality related to system configuration"""
from codecs import open as codecsopen
from collections import abc
from json import dump, load
from pathlib import Path
from typing import Any, Dict, NamedTuple, Tuple

from bcipy.config import DEFAULT_ENCODING


class Parameter(NamedTuple):
    """Represents a single parameter"""
    value: Any
    section: str
    readableName: str
    helpTip: str
    recommended_values: list
    type: str


class ParameterChange(NamedTuple):
    """Represents a Parameter that has been modified from a different value."""
    parameter: Parameter
    original_value: Any


class Parameters(dict):
    """Configuration parameters for BciPy.

        source: str - optional path to a JSON file. If file exists, data will be
            loaded from here. Raises an exception unless the entries are dicts with
            the required_keys.

        cast_values: bool - if True cast values to specified type; default is False.
        """

    def __init__(self, source: str = None, cast_values: bool = False):
        super().__init__()
        self.source = source
        self.cast_values = cast_values

        self.required_keys = set([
            'value', 'section', 'readableName', 'helpTip',
            'recommended_values', 'type'
        ])
        self.conversions = {
            'int': int,
            'float': float,
            'bool': lambda val: val == 'true' or val is True,
            'str': str,
            'directorypath': str,
            'filepath': str
        }
        self.load_from_source()

    @classmethod
    def from_cast_values(cls, **kwargs):
        """Create a new Parameters object from cast values. This is useful
        primarily for testing

        >>> Parameters.from_cast_values(time_prompt=1.0, fake_data=True)
        """
        params = Parameters(source=None, cast_values=True)
        for key, val in kwargs.items():
            value_type = type(val).__name__
            value_str = str(val).lower() if value_type == 'bool' else str(val)
            params.add_entry(
                key, {
                    'value': value_str,
                    'section': '',
                    'readableName': '',
                    'helpTip': '',
                    'recommended_values': '',
                    'type': value_type
                })
        return params

    @property
    def supported_types(self):
        """Supported types for casting values"""
        return self.conversions.keys()

    def cast_value(self, entry: dict):
        """Takes an entry with a desired type and attempts to cast it to that type."""
        cast = self.conversions[entry['type']]
        return cast(entry['value'])

    def serialized_value(self, value, entry_type):
        """Convert a value back into its serialized form"""
        serialized = str(value)
        return serialized.lower() if entry_type == 'bool' else serialized

    def __getitem__(self, key):
        """Override to handle cast values"""
        entry = self.get_entry(key)
        if self.cast_values:
            return self.cast_value(entry)
        return entry

    def __setitem__(self, key, value):
        """Override to handle cast values"""
        if self.cast_values:
            # Can only set values for existing entries when cast.
            entry = self.get_entry(key)
            entry['value'] = self.serialized_value(value, entry['type'])
        else:
            self.add_entry(key, value)

    def add_entry(self, key, value):
        """Adds a configuration parameter."""
        self.check_valid_entry(key, value)
        super().__setitem__(key, value)

    def get_entry(self, key):
        """Get the non-cast entry associated with the given key."""
        return super().__getitem__(key)

    def get(self, key, d=None):
        """Override to handle cast values"""
        entry = super().get(key, d)
        if self.cast_values and entry != d:
            return self.cast_value(entry)
        return entry

    def entries(self):
        """Uncast items"""
        return super().items()

    def items(self):
        """Override to handle cast values"""
        if self.cast_values:
            return [(key, self.cast_value(entry))
                    for key, entry in self.entries()]
        return self.entries()

    def values(self):
        """Override to handle cast values"""
        vals = super().values()
        if self.cast_values:
            return [self.cast_value(entry) for entry in vals]
        return vals

    def update(self, *args, **kwargs):
        """Override to ensure update uses __setitem___"""
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def copy(self):
        """Override
        """
        params = Parameters(source=None, cast_values=self.cast_values)
        params.load(super().copy())
        return params

    def load(self, data: dict):
        """Load values from a dict, validating entries (see check_valid_entry) and raising
        an exception for invalid values.

        data: dict of configuration parameters.
        """
        for name, entry in data.items():
            self.add_entry(name, entry)

    def load_from_source(self):
        """Load data from the configured JSON file."""
        if self.source:
            with codecsopen(self.source, 'r',
                            encoding=DEFAULT_ENCODING) as json_file:
                data = load(json_file)
                self.load(data)

    def check_valid_entry(self, entry_name: str, entry: dict):
        """Checks if the given entry is valid. Raises an exception unless the entry is formatted:

        "fake_data": {
            "value": "true",
            "section": "bci_config",
            "readableName": "Fake Data Sessions",
            "helpTip": "If true, fake data server used",
            "recommended_values": "",
            "type": "bool"
        }

        entry_name : str - name of the configuration parameter
        entry : dict - parameter properties
        """
        if not isinstance(entry, abc.Mapping):
            raise AttributeError(f"'{entry_name}' value must be a dict")
        if set(entry.keys()) != self.required_keys:
            raise Exception(
                f"Incorrect format for key: {entry_name}; value must contain required keys"
            )
        if entry['type'] not in self.supported_types:
            raise Exception(
                f"Type not supported for key: {entry_name}, type: {entry['type']}"
            )
        if entry['type'] == "bool" and entry['value'] not in ['true', 'false']:
            raise Exception(
                f"Invalid value for key: {entry_name}. Must be either 'true' or 'false'"
            )

    def source_location(self) -> Tuple[Path, str]:
        """Location of the source json data if source was provided.

        Returns Tuple(Path, filename: str)
        """
        if self.source:
            path = Path(self.source)
            return (path.parent, path.name)
        return (None, None)

    def save(self, directory: str = None, name: str = None):
        """Save parameters to the given location

        directory: str - optional location to save; default is the source_directory.
        name: str - optional name of new parameters file; default is the source filename.

        Returns the path of the saved file.
        """
        if (not directory or not name) and not self.source:
            raise AttributeError('name and directory parameters are required')

        source_directory, source_name = self.source_location()
        location = directory if directory else source_directory
        filename = name if name else source_name
        path = Path(location, filename)
        with open(path, 'w', encoding=DEFAULT_ENCODING) as json_file:
            dump(dict(self.entries()), json_file, ensure_ascii=False, indent=2)
        return str(path)

    def add_missing_items(self, parameters) -> bool:
        """Given another Parameters instance, add any items that are not already
        present. Existing items will not be updated.

        parameters: Parameters - object from which to add parameters.

        Returns bool indicating whether or not any new items were added.
        """
        updated = False
        existing_keys = self.keys()
        for key, val in parameters.entries():
            if key not in existing_keys:
                self.add_entry(key, val)
                updated = True
        return updated

    def diff(self, parameters) -> Dict[str, ParameterChange]:
        """Lists the differences between this and another set of parameters.
        A None original_value indicates a new parameter.

        Parameters
        ----------
            parameters : Parameters - set of parameters for comparison; these
                are considered the original values and the current set the
                changed values.
        """
        diffs = {}

        for key, param in self.entries():
            if key in parameters.keys():
                original = parameters.get_entry(key)
                if self.cast_value(original) != self.cast_value(param):
                    diffs[key] = ParameterChange(
                        parameter=param, original_value=original['value'])
            else:
                diffs[key] = ParameterChange(parameter=param,
                                             original_value=None)
        return diffs


def changes_from_default(source: str) -> Dict[str, ParameterChange]:
    """Determines which parameters have changed from the default params.

    Parameters
    ----------
        source - path to the parameters json file that will be compared with
            the default parameters.
    """
    default = Parameters(source=DEFAULT_PARAMETERS_PATH, cast_values=True)
    params = Parameters(source=source, cast_values=True)
    return params.diff(default)
