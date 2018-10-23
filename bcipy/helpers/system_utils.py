import sys
import os
from pathlib import Path
import psutil
import pyglet
import importlib
import pkgutil


def get_system_info():

    # Three lines for getting screen resolution
    platform = pyglet.window.get_platform()
    display = platform.get_default_display()
    screen = display.get_default_screen()

    mem = psutil.virtual_memory()

    return {
        'OS': sys.platform,
        'PYTHON': sys.version,
        'RESOLUTION': [screen.width, screen.height],
        'AVAILMEMORYMB': mem.available / 1024 / 1024
    }


def import_submodules(package, recursive=True):
    """ Import all submodules of a module, recursively, including subpackages.
    https://stackoverflow.com/questions/3365740/how-to-import-all-submodules

    Parameters
    ----------
        package : str | package
            name of package or package instance
        recursive : bool, optional

    Returns
    -------
        dict[str, types.ModuleType]
    """
    if type(package) == str or type(package) == unicode:
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        if name.startswith('test'):
            continue
        results[full_name] = importlib.import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results


def dot(relative_to: str, *argv) -> str:
    """
    Given a file location and one or more subdirectory/filename args, returns
    a Path as a str for that file.

    Ex. dot(__file__, 'fst', 'brown_closure.n5.kn.fst')
    """
    working_dir = Path(os.path.dirname(os.path.realpath(relative_to)))
    for subdir in argv:
        # uses the '/' operator in pathlib to construct a new Path.
        working_dir = working_dir / subdir
    return str(working_dir)
