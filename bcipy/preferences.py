"""Module for recording and loading application state and user preferences."""
import json
from pathlib import Path
from typing import Any
from bcipy.config import DEFAULT_ENCODING, PREFERENCES_PATH, BCIPY_ROOT


class Pref:
    """Preference descriptor. When a class attribute is initialized as a Pref,
    values will be stored and retrieved from an 'entries' dict initialized in
    the instance.

    https://docs.python.org/3/howto/descriptor.html

    Parameters
    ----------
        default - default value assigned to the attribute.
    """

    def __init__(self, default: Any = None):
        self.default = default
        self.name = None

    def __set_name__(self, owner, name):
        """Called when the class assigns a Pref to a class attribute."""
        self.name = name

    def __get__(self, instance, owner=None):
        """Retrieve the value from the dict of entries."""
        return instance.entries.get(self.name, self.default)

    def __set__(self, instance, value):
        """Stores the given value in the entries dict keyed on the attribute
        name."""
        instance.entries[self.name] = value
        instance.save()


class Preferences:
    """User preferences persisted to disk to retain application state between
    work sessions.

    Parameters
    ----------
        filename - optional file used for persisting entries.
    """
    signal_model_directory: str = Pref()
    last_directory: str = Pref(default=str(BCIPY_ROOT))

    def __init__(self, filename: str = PREFERENCES_PATH):
        self.filename = filename
        self.entries = {}
        self.load()

    def load(self):
        """Load preference data from the persisted file."""
        if Path(self.filename).is_file():
            with open(self.filename, 'r',
                      encoding=DEFAULT_ENCODING) as json_file:
                for key, val in json.load(json_file).items():
                    self.entries[key] = val

    def save(self):
        """Write preferences to disk."""
        with open(self.filename, 'w', encoding=DEFAULT_ENCODING) as json_file:
            json.dump(self.entries, json_file, ensure_ascii=False, indent=2)

    def get(self, name: str):
        """Get preference by name"""
        return self.entries.get(name, None)

    def set(self, name: str, value: Any, persist: bool = True):
        """Set a preference and save the result.

        Parameters
        ----------
            name - name of the preference
            value - value associated with the given name
            persist - flag indicating whether to immediately save the result.
                Default is True.
        """
        self.entries[name] = value
        if persist:
            self.save()


preferences = Preferences()
