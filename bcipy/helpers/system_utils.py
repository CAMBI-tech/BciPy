import sys
import os

from cpuinfo import get_cpu_info
from typing import Optional, Tuple
from pathlib import Path
import pkg_resources
import platform
import psutil
import pyglet
import importlib
import pkgutil
import logging


def git_dir() -> str:
    """Git Directory.

    Returns the root directory with the .git folder. If this source code
    was not checked out from scm, answers None."""

    # Relative to current file; may need to be modified if method is moved.
    git_root = Path(os.path.abspath(__file__)).parent.parent.parent
    git_meta = Path(os.path.join(git_root, '.git'))
    return os.path.abspath(git_meta) if git_meta.is_dir() else None


def git_hash() -> Optional[str]:
    """Git Hash.

    Returns an abbreviated git sha hash if this code is being run from a
    git cloned version of bcipy; otherwise returns an empty string.

    Could also consider making a system call to:
    git describe --tags
    """

    git_path = git_dir()
    if not git_path:
        print('.git path not found')
        return None

    try:
        head_path = Path(os.path.join(git_path, 'HEAD'))
        with open(head_path) as head_file:
            # First line contains a reference to the current branch.
            # ex. ref: refs/heads/branch_name
            ref = head_file.readline()

        ref_val = ref.split(':')[-1].strip()
        ref_path = Path(os.path.join(git_path, ref_val))
        with open(ref_path) as ref_file:
            sha = ref_file.readline()

        # sha hash is 40 characters; use an abbreviated 7-char version which
        # is displayed in github.
        return sha[0:7]
    except Exception as e:
        print(f'Error reading git version: {e}')
        return None


def bcipy_version() -> str:
    """BciPy Version.

    Gets the current bcipy version. If the current instance of bcipy is a
    git repository, appends the current abbreviated sha hash.
    """
    version = pkg_resources.get_distribution('bcipy').version
    sha_hash = git_hash()

    return f'{version} - {sha_hash}' if sha_hash else version


def get_screen_resolution() -> Tuple[int, int]:
    """Gets the screen resolution.

    Note: Use this method if only the screen resolution is needed; it is much more efficient
    than extracting that information from the dict returned by the get_system_info method.

    Returns
    -------
        (width, height)
     """
    screen = pyglet.canvas.get_display().get_default_screen()
    return (screen.width, screen.height)


def get_system_info() -> dict:
    """Get System Information.
    See: https://stackoverflow.com/questions/3103178/how-to-get-the-system-info-with-python

    Returns
    -------
        dict of system-related properties, including ['os', 'py_version', 'resolution', 
          'memory', 'bcipy_version', 'platform', 'platform-release', 'platform-version', 
          'architecture', 'processor', 'cpu_count', 'hz', 'ram']
    """
    screen_width, screen_height = get_screen_resolution()
    return {
        'os': sys.platform,
        'py_version': sys.version,
        'resolution': [screen_width, screen_height],
        'memory': psutil.virtual_memory().available / 1024 / 1024,
        'bcipy_version': bcipy_version(),
        'platform': platform.system(),
        'platform-release': platform.release(),
        'platform-version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'hz': get_cpu_info()['hz_actual_friendly'],
        'ram': str(round(psutil.virtual_memory().total / (1024.0**3))) + " GB"
    }


def configure_logger(
        save_folder: str,
        log_name='bcipy_session.txt',
        log_level=logging.DEBUG,
        version=None) -> None:
    """Configure Logger.

    Does what it says.
    """
    # create the log file
    logfile = os.path.join(save_folder, 'logs', log_name)

    # configure it
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    handler = logging.FileHandler(logfile, 'w', encoding='utf-8')
    handler.setFormatter(logging.Formatter('(%(threadName)-9s) %(message)s'))
    root_logger.addHandler(handler)

    print(f'Printing all BciPy logs to: {logfile}')

    if version:
        logging.info(f'Start of Session for BciPy Version: ({version})')


def import_submodules(package, recursive=True):
    """Import Submodules.

    Import all submodules of a module, recursively, including subpackages.
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
    if isinstance(package, str) or isinstance(package, unicode):
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


def auto_str(cls):
    """Autogenerate a str method to print all variable names and values.
    https://stackoverflow.com/questions/32910096/is-there-a-way-to-auto-generate-a-str-implementation-in-python
    """

    def __str__(self):
        return '%s(%s)' % (type(self).__name__, ', '.join(
            '%s=%s' % item for item in vars(self).items()))

    cls.__str__ = __str__
    return cls
