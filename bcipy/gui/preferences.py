"""Module for recording and loading application state and user preferences."""
import json
from pathlib import Path
from typing import Any, Callable
from bcipy.config import DEFAULT_ENCODING, PREFERENCES_PATH


class Preferences:
    """User preferences persisted to disk to retain application state between
    work sessions.

    Parameters
    ----------
        filename - optional file used for persisting entries.
    """

    # Default values.
    # Properties included here will have instance methods for getting and setting.
    DEFAULTS = {'signal_model_directory': None, 'last_directory': None}

    def __init__(self, filename: str = PREFERENCES_PATH):
        self.filename = filename
        self.entries = Preferences.DEFAULTS
        self.load()

    def __getattr__(self, name):
        ''' will only get called for undefined attributes '''
        raise Exception(f'No "{name}" member for preferences')

    def load(self):
        """Load preference data from the persisted file."""
        if Path(self.filename).is_file():
            with open(self.filename, 'r',
                      encoding=DEFAULT_ENCODING) as json_file:
                self.entries = json.load(json_file)

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


def get_pref_method(pref_name: str) -> Callable:
    """Creates an instance method for getting a preference by name."""

    def get_pref(self):
        return self.get(pref_name)

    return get_pref


def set_pref_method(pref_name) -> Callable:
    """Creates an instance method for setting a preference."""

    def set_pref(self, value: Any, persist: bool = True):
        return self.set(pref_name, value, persist)

    return set_pref


def add_dynamic_methods():
    """Dynamically create methods for setting known properties."""
    for name in Preferences.DEFAULTS:
        setattr(Preferences, f'get_{name}', get_pref_method(name))
        setattr(Preferences, f'set_{name}', set_pref_method(name))


add_dynamic_methods()
preferences = Preferences()
