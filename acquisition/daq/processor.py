"""DAQ Item Processors"""

from __future__ import absolute_import, division, print_function

import csv


class Processor(object):
    """Abstract base class for an object that can be used to process data
    acquisition data.

    Parameters
    ----------
        device_name : str
            Name of the device used to collect data.
        hz : int
            Sample frequency in Hz.
        channels : list
            List of channel names.
    """

    def __init__(self, device_name, hz, channels):
        self._device_name = device_name
        self._hz = hz
        self._channels = channels

    def __enter__(self):
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
    """A DAQ item Processor that writes items to a file."""

    def __init__(self, filename, device_name, hz, channels):
        super(FileWriter, self).__init__(device_name, hz, channels)
        self._filename = filename
        self._writer = None

    @classmethod
    def builder(cls, filename):
        """Returns a builder than constructs a new FileWriter with the given
        filename."""
        def build(device_name, hz, channels):
            return FileWriter(filename, device_name, hz, channels)
        return build

    # @override ; context manager
    def __enter__(self):
        self._file = open(self._filename, 'w')
        self._writer = csv.writer(self._file, delimiter=',')
        self._writer.writerow(['daq_type', self._device_name])
        self._writer.writerow(['sample_rate', self._hz])
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

    def __init__(self, device_name, hz, channels):
        import pylsl
        import uuid

        super(LslProcessor, self).__init__(device_name, hz, channels)
        info = pylsl.StreamInfo(device_name, 'EEG', len(channels), hz,
                                'float32', str(uuid.uuid4()))
        meta_channels = info.desc().append_child('channels')
        for c in channels:
            meta_channels.append_child('channel') \
                .append_child_value('label', str(c)) \
                .append_child_value('unit', 'microvolts') \
                .append_child_value('type', 'EEG')

        self._outlet = pylsl.StreamOutlet(info)

    def process(self, record, timestamp=None):
        self._outlet.push_sample(record, timestamp)
