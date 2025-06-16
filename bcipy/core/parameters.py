"""Module for functionality related to system configuration"""
from codecs import open as codecsopen
from collections import abc
from json import dump, load
from pathlib import Path
from re import fullmatch
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union, List

from bcipy.config import DEFAULT_ENCODING, DEFAULT_PARAMETERS_PATH


class Parameter(NamedTuple):
    """Represents a single parameter."

    Attributes:
        value (Any): The value of the parameter.
        section (str): The section the parameter belongs to.
        name (str): The display name of the parameter.
        helpTip (str): A helpful tip or description for the parameter.
        recommended (list): Recommended values for the parameter.
        editable (bool): Whether the parameter is editable.
        type (str): The data type of the parameter (e.g., 'int', 'float', 'bool', 'str', 'range').
    """
    value: Any
    section: str
    name: str
    helpTip: str
    recommended: list
    editable: bool
    type: str


class ParameterChange(NamedTuple):
    """Represents a Parameter that has been modified from a different value."

    Attributes:
        parameter (Parameter): The modified parameter.
        original_value (Any): The original value of the parameter before modification.
    """
    parameter: Union[Parameter, dict]
    original_value: Any


def parse_range(range_str: str) -> Tuple[Union[int, float], Union[int, float]]:
    """Parses the range description into a tuple of (low, high).

    If either value can be parsed as a float, the resulting tuple will have
    float values; otherwise, they will be integers.

    Args:
        range_str (str): Range description formatted as 'low:high'.

    Returns:
        Tuple[Union[int, float], Union[int, float]]: A tuple containing the low and high values of the range.

    Raises:
        AssertionError: If the `range_str` is not in the format 'low:high' or if the low value is not less than the high value.

    Examples:
        >>> parse_range("1:10")
        (1, 10)
    """
    assert ':' in range_str, "Invalid range format; values must be separated by ':'"
    low_str, high_str = range_str.split(':')
    int_pattern = "-?\\d+"
    if fullmatch(int_pattern, low_str) and fullmatch(int_pattern, high_str):
        low: Union[int, float] = int(low_str)
        high: Union[int, float] = int(high_str)
    else:
        low = float(low_str)
        high = float(high_str)

    assert low < high, "Low value must be less that the high value"
    return (low, high)


def serialize_value(value_type: str, value: Any) -> str:
    """Serializes the given value to a string.

    Serialized values should be able to be cast using the `conversions` dictionary
    defined in the `Parameters` class.

    Args:
        value_type (str): The declared type of the value (e.g., 'bool', 'range').
        value (Any): The value to serialize.

    Returns:
        str: The serialized string representation of the value.
    """
    if value_type == 'bool':
        return str(value).lower()
    if value_type == 'range':
        low, high = value
        return f"{low}:{high}"
    return str(value)


class Parameters(dict):
    """Configuration parameters for BciPy.

    This class extends `dict` to provide type-casting and validation for
    configuration parameters, typically loaded from a JSON file.

    Args:
        source (Optional[str], optional): Optional path to a JSON file. If the file exists,
            data will be loaded from here. Raises an exception unless the entries are
            dictionaries with the required keys. Defaults to None.
        cast_values (bool, optional): If True, values will be cast to their specified type
            when accessed. Defaults to False.
    """

    def __init__(self, source: Optional[str] = None, cast_values: bool = False):
        super().__init__()
        self.source = source
        self.cast_values = cast_values

        self.required_keys = set(Parameter._fields)
        self.conversions = {
            'int': int,
            'float': float,
            'bool': lambda val: val == 'true' or val is True,
            'str': str,
            'directorypath': str,
            'filepath': str,
            'range': parse_range
        }
        self.load_from_source()

    @classmethod
    def from_cast_values(cls, **kwargs: Any) -> 'Parameters':
        """Creates a new `Parameters` object from cast values.

        This is useful primarily for testing.

        Args:
            **kwargs (Any): Keyword arguments representing parameter names and their values.

        Returns:
            Parameters: A new `Parameters` instance with values cast.

        Examples:
            >>> Parameters.from_cast_values(time_prompt=1.0, fake_data=True)
        """
        params = Parameters(source=None, cast_values=True)
        for key, val in kwargs.items():
            value_type = type(val).__name__
            # Convert tuple to range
            if value_type == 'tuple':
                value_type = 'range'
            value_str = serialize_value(value_type, val)
            params.add_entry(
                key, {
                    'value': value_str,
                    'section': '',
                    'name': '',
                    'helpTip': '',
                    'recommended': '',
                    'editable': '',
                    'type': value_type
                })
        return params

    @property
    def supported_types(self) -> list:
        """Returns the set of supported types for casting values."""
        return list(self.conversions.keys())

    def cast_value(self, entry: Dict[str, Any]) -> Any:
        """Takes an entry with a desired type and attempts to cast it to that type.

        Args:
            entry (Dict[str, Any]): A dictionary representing a parameter entry with a 'type' key.

        Returns:
            Any: The value cast to the specified type.
        """
        cast = self.conversions[entry['type']]
        return cast(entry['value'])  # type: ignore

    def serialized_value(self, value: Any, entry_type: str) -> str:
        """Converts a value back into its serialized string form."

        Args:
            value (Any): The value to serialize.
            entry_type (str): The declared type of the value.

        Returns:
            str: The serialized string representation of the value.
        """
        serialized = str(value)
        return serialized.lower() if entry_type == 'bool' else serialized

    def __getitem__(self, key: str) -> Any:
        """Overrides dictionary item access to handle cast values."

        Args:
            key (str): The key of the parameter to retrieve.

        Returns:
            Any: The cast value of the parameter if `cast_values` is True, otherwise the raw entry dictionary.
        """
        entry = self.get_entry(key)
        if self.cast_values:
            return self.cast_value(entry)
        return entry

    def __setitem__(self, key: str, value: Any) -> None:
        """Overrides dictionary item assignment to handle cast values and validate entries."

        If `cast_values` is True, it attempts to set the serialized value for an existing entry.
        Otherwise, it adds a new entry after validation.

        Args:
            key (str): The key of the parameter to set.
            value (Any): The value to set for the parameter.
        """
        if self.cast_values:
            # Can only set values for existing entries when cast.
            entry = self.get_entry(key)
            entry['value'] = self.serialized_value(value, entry['type'])
        else:
            self.add_entry(key, value)

    def add_entry(self, key: str, value: Dict[str, Any]) -> None:
        """Adds a configuration parameter after validating its format."

        Args:
            key (str): The name of the configuration parameter.
            value (Dict[str, Any]): A dictionary containing the parameter properties.
        """
        self.check_valid_entry(key, value)
        super().__setitem__(key, value)

    def get_entry(self, key: str) -> Dict[str, Any]:
        """Gets the non-cast entry associated with the given key."

        Args:
            key (str): The key of the parameter entry to retrieve.

        Returns:
            Dict[str, Any]: The raw dictionary entry for the parameter.
        """
        return super().__getitem__(key)

    def get(self, key: str, d: Optional[Any] = None) -> Any:
        """Overrides dictionary `get` method to handle cast values."

        Args:
            key (str): The key of the parameter to retrieve.
            d (Optional[Any], optional): Default value to return if the key is not found.
                Defaults to None.

        Returns:
            Any: The cast value of the parameter if `cast_values` is True and the key is found,
                 otherwise the raw entry or the default value.
        """
        entry = super().get(key, d)
        if self.cast_values and entry != d:
            return self.cast_value(entry)
        return entry

    def entries(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Returns the uncast items (key-value pairs) of the parameters.

        Returns:
            List[Tuple[str, Dict[str, Any]]]: A list of key-value tuples, where values are raw entry dictionaries.
        """
        return list(super().items())  # type: ignore

    def items(self) -> List[Tuple[str, Any]]:  # type: ignore
        """Overrides dictionary `items` method to handle cast values.

        Returns:
            List[Tuple[str, Any]]: A list of key-value tuples, where values are cast if `cast_values` is True.
        """
        if self.cast_values:
            return [(key, self.cast_value(entry))
                    for key, entry in self.entries()]
        return self.entries()

    def values(self) -> List[Any]:  # type: ignore
        """Override to handle cast values.

        Returns:
            List[Any]: A list of parameter values, cast if `cast_values` is True.
        """
        vals = super().values()
        if self.cast_values:
            return [self.cast_value(entry) for entry in vals]
        return list(vals)  # type: ignore

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Overrides dictionary `update` method to ensure `__setitem__` is used for each item."

        Args:
            *args (Any): Positional arguments for dictionary update.
            **kwargs (Any): Keyword arguments for dictionary update.
        """
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def copy(self) -> 'Parameters':
        """Creates a shallow copy of the `Parameters` object."

        Returns:
            Parameters: A new `Parameters` instance with the same parameters.
        """
        params = Parameters(source=None, cast_values=self.cast_values)
        params.load(super().copy()) # type: ignore
        return params

    def load(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Loads values from a dictionary, validating entries and raising an exception for invalid values."

        Args:
            data (Dict[str, Dict[str, Any]]): A dictionary of configuration parameters.
        """
        for name, entry in data.items():
            self.add_entry(name, entry)

    def load_from_source(self) -> None:
        """Loads data from the configured JSON file."

        If `self.source` is set, it attempts to open and load the JSON data from that path.
        """
        if self.source:
            with codecsopen(self.source, 'r',
                            encoding=DEFAULT_ENCODING) as json_file:
                data = load(json_file)
                self.load(data) # type: ignore

    def check_valid_entry(self, entry_name: str, entry: Dict[str, Any]) -> None:
        """Checks if the given entry is valid. Raises an exception unless the entry is formatted as expected."

        Expected format:
        ```json
        "fake_data": {
            "value": "true",
            "section": "bci_config",
            "name": "Fake Data Sessions",
            "helpTip": "If true, fake data server used",
            "recommended": "",
            "editable": true,
            "type": "bool"
        }
        ```

        Args:
            entry_name (str): Name of the configuration parameter.
            entry (Dict[str, Any]): Parameter properties.

        Raises:
            AttributeError: If `entry` is not a dictionary.
            Exception: If `entry` does not contain required keys, if the 'type' is not supported,
                       or if the 'value' for a 'bool' type is invalid.
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

    def source_location(self) -> Tuple[Optional[Path], Optional[str]]:
        """Returns the location of the source JSON data if a source was provided."

        Returns:
            Tuple[Optional[Path], Optional[str]]: A tuple containing the parent directory path and the filename.
        """
        if self.source:
            path = Path(self.source)
            return (path.parent, path.name)
        return (None, None)

    def save(self, directory: Optional[str] = None, name: Optional[str] = None) -> str:
        """Saves parameters to the given location."

        Args:
            directory (Optional[str], optional): Optional location to save the file. Defaults to the source directory.
            name (Optional[str], optional): Optional name of the new parameters file. Defaults to the source filename.

        Returns:
            str: The path of the saved file.

        Raises:
            AttributeError: If neither `directory` and `name` are provided nor a `source` path is set.
        """
        if (not directory or not name) and not self.source:
            raise AttributeError('name and directory parameters are required')

        source_directory, source_name = self.source_location()
        location = directory if directory else source_directory
        filename = name if name else source_name
        path = Path(location, filename) # type: ignore
        with open(path, 'w', encoding=DEFAULT_ENCODING) as json_file:
            dump(dict(self.entries()), json_file, ensure_ascii=False, indent=2) # type: ignore
        return str(path)

    def add_missing_items(self, parameters: 'Parameters') -> bool:
        """Given another `Parameters` instance, adds any items that are not already present.

        Existing items will not be updated.

        Args:
            parameters (Parameters): Object from which to add parameters.

        Returns:
            bool: True if any new items were added, False otherwise.
        """
        updated = False
        existing_keys = self.keys()
        for key, val in parameters.entries():
            if key not in existing_keys:
                self.add_entry(key, val)
                updated = True
        return updated

    def diff(self, parameters: 'Parameters') -> Dict[str, ParameterChange]:
        """Lists the differences between this and another set of parameters.
        A None original_value indicates a new parameter.

        Args:
            parameters (Parameters): Set of parameters for comparison; these
                are considered the original values and the current set the
                changed values.

        Returns:
            Dict[str, ParameterChange]: A dictionary where keys are parameter names
                and values are ParameterChange objects.
        """
        diffs: Dict[str, ParameterChange] = {}

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

    def instantiate(self, named_tuple_class: type[NamedTuple]) -> NamedTuple:
        """Instantiates a `NamedTuple` whose fields represent a subset of the parameters."

        Args:
            named_tuple_class (type[NamedTuple]): The `NamedTuple` class to instantiate.

        Returns:
            NamedTuple: An instance of the provided `NamedTuple` class with values populated from parameters.
        """
        vals = [
            self.cast_value(self.get_entry(key))
            for key in named_tuple_class._fields
        ]
        return named_tuple_class(*vals)


def changes_from_default(source: str) -> Dict[str, ParameterChange]:
    """Determines which parameters have changed from the default parameters.

    Args:
        source (str): Path to the parameters JSON file that will be compared with
            the default parameters.

    Returns:
        Dict[str, ParameterChange]: A dictionary of `ParameterChange` objects representing the differences.
    """
    default = Parameters(source=DEFAULT_PARAMETERS_PATH, cast_values=True)
    params = Parameters(source=source, cast_values=True)
    return params.diff(default)
