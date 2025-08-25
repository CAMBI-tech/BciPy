class DeviceSpecNotFoundError(Exception):
    """Thrown when a suitable DeviceSpec was not found in the devices.json file."""


class IncompatibleData(Exception):
    """Thrown when data is not suitable for the current simulation."""


class IncompatibleDeviceSpec(IncompatibleData):
    """Thrown when a DeviceSpec in a data directory did not match the data or
    when the model's spec was incompatible.
    """


class IncompatibleParameters(IncompatibleData):
    """Thrown when the timing parameters used for data collection are
    incompatible with the timing parameters of the simulation.
    """


class IncompatibleSampler(Exception):
    """Thrown when the provided sampler is incompatible with a given task."""
