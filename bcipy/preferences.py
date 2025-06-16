"""Module for recording and loading application state and user preferences.

This module provides functionality for storing and retrieving user preferences
and application state between sessions using a JSON-based storage system.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Type

from bcipy.config import BCIPY_ROOT, DEFAULT_ENCODING, PREFERENCES_PATH


class Pref:
    """A descriptor class for managing preferences.

    When a class attribute is initialized as a Pref, values will be stored and
    retrieved from an 'entries' dict initialized in the instance.

    For more information on descriptors, see:
    https://docs.python.org/3/howto/descriptor.html

    Args:
        default: Default value assigned to the attribute if not found in entries.
    """

    def __init__(self, default: Optional[Any] = None) -> None:
        self.default = default
        self.name: Optional[str] = None

    def __set_name__(self, owner: Type[Any], name: str) -> None:
        """Called when the class assigns a Pref to a class attribute.

        Args:
            owner: The class that owns this descriptor.
            name: The name of the attribute this descriptor is assigned to.
        """
        self.name = name

    def __get__(self, instance: Any, owner: Optional[Type[Any]] = None) -> Any:
        """Retrieve the value from the dict of entries.

        Args:
            instance: The instance that this descriptor is accessed from.
            owner: The class that owns this descriptor.

        Returns:
            The value stored in entries or the default value.
        """
        if instance is None:
            return self
        return instance.entries.get(self.name, self.default)

    def __set__(self, instance: Any, value: Any) -> None:
        """Store the given value in the entries dict keyed on the attribute name.

        Args:
            instance: The instance that this descriptor is accessed from.
            value: The value to store in the entries dict.
        """
        instance.entries[self.name] = value
        instance.save()


class Preferences:
    """User preferences persisted to disk to retain application state between work sessions.

    This class manages user preferences by storing them in a JSON file on disk. It provides
    methods for loading, saving, and accessing preference values.

    Attributes:
        signal_model_directory: Directory containing signal models.
        last_directory: Last accessed directory, defaults to BCIPY_ROOT.

    Args:
        filename: Optional file used for persisting entries.
    """
    signal_model_directory = Pref()
    last_directory = Pref(default=str(BCIPY_ROOT))

    def __init__(self, filename: str = PREFERENCES_PATH) -> None:
        self.filename = filename
        self.entries: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load preference data from the persisted file.

        Reads the JSON file specified by self.filename and populates the entries
        dictionary with the stored preferences.
        """
        if Path(self.filename).is_file():
            with open(self.filename, 'r',
                      encoding=DEFAULT_ENCODING) as json_file:
                for key, val in json.load(json_file).items():
                    self.entries[key] = val

    def save(self) -> None:
        """Write preferences to disk.

        Saves the current entries dictionary to the JSON file specified by
        self.filename.
        """
        with open(self.filename, 'w', encoding=DEFAULT_ENCODING) as json_file:
            json.dump(self.entries, json_file, ensure_ascii=False, indent=2)

    def get(self, name: str) -> Optional[Any]:
        """Get preference by name.

        Args:
            name: Name of the preference to retrieve.

        Returns:
            The preference value if found, None otherwise.
        """
        return self.entries.get(name, None)

    def set(self, name: str, value: Any, persist: bool = True) -> None:
        """Set a preference and save the result.

        Args:
            name: Name of the preference.
            value: Value associated with the given name.
            persist: Flag indicating whether to immediately save the result.
                Default is True.
        """
        self.entries[name] = value
        if persist:
            self.save()


preferences = Preferences()
