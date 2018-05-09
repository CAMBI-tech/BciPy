"""Used to find a protocol or device by name."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from utils.pypaths import import_submodules
from acquisition.protocols.device import Device
from acquisition.protocols.dsi.dsi_protocol import DsiProtocol

# import all submodules so we can introspect on subclasses.
import_submodules('protocols')


def _key(pyclass):
    """
    >>> from acquisition.protocols.dsi_device import DsiDevice
    >>> _key(DsiDevice)
    u'DSI'
    """
    return pyclass.__module__.split(".")[-1].split('_')[0].upper()


supported_devices = dict((_key(device), device)
                         for device in Device.__subclasses__())


# TODO: Refactor protocols to use a base class so we can introspect on them.
# Otherwise, consider using a naming convention.
supported_protocols = {
    'DSI': DsiProtocol
}


def find_device(name):
    """Find device by name.

    Parameters
    ----------
        name : str
            name of the device
    Returns
    -------
        Device constructor
    """
    return supported_devices[name]


def protocol_with(name, fs, channels):
    """Find protocol by name and initialize with the given parameters.

    Parameters
    ----------
        name : str
            name of the device
        fs : int
            used to override the default protocol fs value
        channels : list
            used to override the default protocol channels value
    Returns
    -------
        Protocol
    """
    return supported_protocols[name](fs, channels)


def default_protocol(name):
    """Find protocol by name and initialize with default options.

    Parameters
    ----------
        name : str
            name of the device
    Returns
    -------
        Protocol
    """
    return supported_protocols[name]()
