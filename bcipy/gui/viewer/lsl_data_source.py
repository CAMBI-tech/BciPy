"""Streams data from pylsl and puts it into a Queue."""
import pylsl
from bcipy.acquisition.util import StoppableThread
from bcipy.acquisition.device_info import DeviceInfo
from bcipy.gui.viewer.data_source import DataSource


class LslDataSource(DataSource):
    """DataSource that rads from a pylsl StreamInlet.

    Parameters
    ----------
        stream_type: str 
            StreamInlet stream type; default is 'EEG'
    """

    def __init__(self, stream_type: str = 'EEG'):
        super(LslDataSource, self).__init__()

        print("Waiting for LSL EEG data stream...")
        self.stream_type = stream_type
        streams = pylsl.resolve_stream('type', self.stream_type)
        inlet = pylsl.StreamInlet(streams[0])
        info = inlet.info()

        fs = float(info.nominal_srate())
        name = info.name()
        channel_names = []
        ch = info.desc().child("channels").child("channel")
        for k in range(info.channel_count()):
            channel_names.append(ch.child_value("label"))
            ch = ch.next_sibling()

        self.device_info = DeviceInfo(fs=fs, channels=channel_names, name=name)
        self.inlet = inlet

    def next(self):
        """Provide the next record."""
        sample, _ts = self.inlet.pull_sample(timeout=0.0)
        return sample

    def next_n(self, n: int):
        """Provides the next n records as a list"""
        # Read data from the inlet. Use a timeout of 0.0 so we don't block GUI interaction.
        samples, _ts = self.inlet.pull_chunk(timeout=0.0, max_samples=n)

        if len(samples) < n:
            raise StopIteration
        return samples
