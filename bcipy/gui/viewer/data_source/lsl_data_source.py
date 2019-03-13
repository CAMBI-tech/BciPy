"""Streams data from pylsl and puts it into a Queue."""
import pylsl
from bcipy.acquisition.util import StoppableThread
from bcipy.acquisition.device_info import DeviceInfo
from bcipy.gui.viewer.data_source.data_source import DataSource


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
        self.sample_rate = fs
        print(f"Sample rate: {fs}")
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

    def next_n(self, n: int, fast_forward=False):
        """Provides the next n records as a list"""
        # Read data from the inlet. May blocks GUI interaction if n samples
        # are not yet available.
        samples, _ts = self.inlet.pull_chunk(timeout=0.1, max_samples=n)
        
        if fast_forward:
            tmp = samples
            print(f"Fast forwarding:")
            chomped_count = 0
            while len(tmp) == n:
                samples = tmp
                # A timeout of 0.0 does not block GUI interaction and only
                # gets samples immediately available.
                tmp, _ts = self.inlet.pull_chunk(timeout=0.0, max_samples=n)
                chomped_count += len(tmp)
            print(f"Chomped {chomped_count} records.")
            samples = samples[len(tmp):] + tmp            

        if len(samples) < n:
            raise StopIteration
        return samples
        