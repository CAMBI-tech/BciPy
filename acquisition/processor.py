"""DAQ Item Processors"""

from __future__ import absolute_import, division, print_function

import csv
import sys


class Processor(object):
    """Abstract base class for an object that can be used to process data
    acquisition data.
    """

    def set_device_info(self, device_name, fs, channels):
        """
        Sets the device info, which may be used by the process method.

        Parameters
        ----------
            device_name : str
                Name of the device used to collect data.
            fs : int
                Sample frequency in Hz.
            channels : list
                List of channel names.
        """
        self._device_name = device_name
        self._fs = fs
        self._channels = channels

    def _check_device_info(self):
        """
        Checks that the device_info has been set.

        Exceptions
        ----------
            throws AssertionError unless all device_info properties are set.
        """
        assert self._device_name is not None and \
            self._fs is not None and \
            self._channels is not None, \
            "device_info is not set; Initialize with set_device_info."

    def __enter__(self):
        # Device info should be set before using as a context.
        self._check_device_info()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def process(self, record, timestamp=None):
        """Process the given record.

        Parameters
        ----------
            record: list<float>
                A single record returned by from reading a data packet with a
                float value for each channel.
            timestamp: float, optional
                time at which the record was obtained
        """
        raise NotImplementedError('Subclass must define the process method')


class FileWriter(Processor):
    """A DAQ item Processor that writes items to a file.

    Parameters
    ----------
        filename : str
            Filename to write to.
    """

    def __init__(self, filename):
        super(FileWriter, self).__init__()
        self._filename = filename
        self._writer = None

    # @override ; context manager
    def __enter__(self):
        self._check_device_info()

        # For python 2, writer needs the 'wb' option in order to work on
        # Windows. If using #Python3 'w' is needed.
        if sys.version_info >= (3, 0, 0):
            self._file = open(self._filename, 'w', newline='')
        else:
            self._file = open(self._filename, 'wb')

        self._writer = csv.writer(self._file, delimiter=',')
        self._writer.writerow(['daq_type', self._device_name])
        self._writer.writerow(['sample_rate', self._fs])
        self._writer.writerow(['timestamp'] + self._channels)
        return self

    # @override ; context manager
    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()

    def process(self, record, timestamp=None):
        if self._writer:
            self._writer.writerow([timestamp] + record)


class LslProcessor(Processor):
    """A DAQ item processor that writes to an LSL data stream."""

    def __init__(self, filename):
        super(LslProcessor, self).__init__()
        self._outlet = None

    # @override
    def set_device_info(self, device_name, fs, channels):
        import pylsl
        import uuid

        super(LslProcessor, self).set_device_info(device_name, fs, channels)
        info = pylsl.StreamInfo(device_name, 'EEG', len(channels), fs,
                                'float32', str(uuid.uuid4()))
        meta_channels = info.desc().append_child('channels')
        for c in channels:
            meta_channels.append_child('channel') \
                .append_child_value('label', str(c)) \
                .append_child_value('unit', 'microvolts') \
                .append_child_value('type', 'EEG')

        self._outlet = pylsl.StreamOutlet(info)

    def process(self, record, timestamp=None):
        if self._outlet:
            self._outlet.push_sample(record, timestamp)
