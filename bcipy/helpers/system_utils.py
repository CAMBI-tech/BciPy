# mypy: disable-error-code="import-untyped"
"""Utilities for system information and general functionality that may be
shared across modules."""
import importlib
import logging
import os
import pkgutil
import platform
import socket
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional

import pkg_resources
import psutil
import pyglet
import torch
from cpuinfo import get_cpu_info

from bcipy.config import DEFAULT_ENCODING, SESSION_LOG_FILENAME


class ScreenInfo(NamedTuple):
    width: int
    height: int
    rate: float


def is_connected(hostname: str = "1.1.1.1", port=80) -> bool:
    """Test for internet connectivity.

    Parameters
    ----------
        hostname - name of host for attempted connection
        port - port on host to which to connect
    """
    try:
        conn = socket.create_connection(address=(hostname, port), timeout=2)
        conn.close()
        return True
    except OSError:
        pass
    return False


def is_battery_powered() -> bool:
    """Check if this current computer is a laptop currently using its battery.

    Returns
    -------
    True if the computer is currently running on battery power. This can impact
    the performance of hardware (ex. GPU) needed for BciPy operation by entering
    power saving operations."""
    return psutil.sensors_battery(
    ) and not psutil.sensors_battery().power_plugged


def is_screen_refresh_rate_low(refresh: Optional[float] = None) -> bool:
    """Check if the screen refresh rate is sufficient for BciPy operation.

    Parameters
    ----------
        refresh(float) - optional screen refresh rate if already collected. If not provided,
            the current screen refresh rate is gathered using get_screen_info().

    Returns
    -------
    True if the refresh rate is less than 120 Hz; otherwise False.
    """
    if refresh is None:
        refresh = get_screen_info().rate

    return refresh < 120


def git_dir() -> Optional[str]:
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
        with open(head_path, encoding=DEFAULT_ENCODING) as head_file:
            # First line contains a reference to the current branch.
            # ex. ref: refs/heads/branch_name
            ref = head_file.readline()

        ref_val = ref.split(':')[-1].strip()
        ref_path = Path(os.path.join(git_path, ref_val))
        with open(ref_path, encoding=DEFAULT_ENCODING) as ref_file:
            sha = ref_file.readline()

        # sha hash is 40 characters; use an abbreviated 7-char version which
        # is displayed in github.
        return sha[0:7]
    except Exception as error:
        print(f'Error reading git version: {error}')
        return None


def bcipy_version() -> str:
    """BciPy Version.

    Gets the current bcipy version. If the current instance of bcipy is a
    git repository, appends the current abbreviated sha hash.
    """
    version: str = pkg_resources.get_distribution('bcipy').version
    sha_hash = git_hash()

    return f'{version} - {sha_hash}' if sha_hash else version


def get_screen_info(stim_screen: Optional[int] = None) -> ScreenInfo:
    """Gets the screen information.

    Note: Use this method if only the screen resolution is needed; it is much more efficient
    than extracting that information from the dict returned by the get_system_info method.

    Returns
    -------
        ScreenInfo(width, height, rate)
     """
    if stim_screen:
        screen = pyglet.canvas.get_display().get_screens()[stim_screen]
    else:
        screen = pyglet.canvas.get_display().get_default_screen()
    return ScreenInfo(screen.width, screen.height, screen.get_mode().rate)


def get_gpu_info() -> List[dict]:
    """Information about GPUs available for processing."""
    properties = []
    for idx in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(idx)
        properties.append(dict(name=prop.name, total_memory=prop.total_memory))
    return properties


def get_system_info() -> dict:
    """Get System Information.
    See: https://stackoverflow.com/questions/3103178/how-to-get-the-system-info-with-python

    Returns
    -------
        dict of system-related properties, including ['os', 'py_version', 'resolution',
          'memory', 'bcipy_version', 'platform', 'platform-release', 'platform-version',
          'architecture', 'processor', 'cpu_count', 'hz', 'ram']
    """
    screen_info = get_screen_info()
    info = get_cpu_info()
    gpu_info = get_gpu_info()

    return {
        'os': sys.platform,
        'py_version': sys.version,
        'screen_resolution': [screen_info.width, screen_info.height],
        'memory': psutil.virtual_memory().available / 1024 / 1024,
        'bcipy_version': bcipy_version(),
        'platform': platform.system(),
        'platform-release': platform.release(),
        'platform-version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'cpu_brand': info['brand_raw'],
        'gpu_available': torch.cuda.is_available(),
        'gpu_count': len(gpu_info),
        'gpu_details': gpu_info,
        'screen_refresh': f"{screen_info.rate}Hz",
        'ram': str(round(psutil.virtual_memory().total / (1024.0**3))) + " GB"
    }


def configure_logger(
        save_folder: str,
        log_name=SESSION_LOG_FILENAME,
        log_level=logging.INFO) -> logging.Logger:
    """Configure Logger.

    Configures a logger to print to a file in the save_folder directory.
    """
    # create the log file
    logfile = os.path.join(save_folder, 'logs', log_name)

    # configure it
    logger = logging.getLogger(name=log_name)
    logger.setLevel(log_level)
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    handler = logging.FileHandler(logfile, 'w', encoding='utf-8')
    handler.setFormatter(logging.Formatter(
        '[%(threadName)-9s][%(asctime)s][%(name)s][%(levelname)s]: %(message)s'))
    logger.addHandler(handler)

    # print to console the absolute path of the log file to aid in debugging
    path_to_logs = os.path.abspath(logfile)
    msg = f'Writing logs to: {path_to_logs}'
    logger.info(msg)
    print(msg)

    return logger


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
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for _loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
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


def log_to_stdout():
    """Set logging to stdout. Useful for demo scripts.
    https://stackoverflow.com/questions/14058453/making-python-loggers-output-all-messages-to-stdout-in-addition-to-log-file
    """

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(threadName)-9s][%(asctime)s][%(name)s][%(levelname)s]: %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)


def report_execution_time(func: Callable) -> Callable:
    """Report execution time.

    A decorator to log execution time of methods in seconds. To use,
        decorate your method with @report_execution_time.
    """
    log = logging.getLogger()

    def wrap(*args, **kwargs):
        time1 = time.perf_counter()
        response = func(*args, **kwargs)
        time2 = time.perf_counter()
        log.info('{:s} method took {:0.4f}s to execute'.format(func.__name__, (time2 - time1)))
        return response
    return wrap


class AutoNumberEnum(Enum):
    """Enum that auto-numbers the value for each item.

    See: https://docs.python.org/3/howto/enum.html
    """

    def __new__(cls, *args):
        """Autoincrements the value of each item added to the enum."""
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj
