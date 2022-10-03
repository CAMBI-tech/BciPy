"""Streams data from pylsl and puts it into a Queue."""
import pylsl
from bcipy.acquisition.protocols.lsl.lsl_client import device_from_metadata
from bcipy.gui.viewer.data_source.data_source import DataSource


class LslDataSource(DataSource):
    """DataSource that provides data from an underlying pylsl StreamInlet.

    Parameters
    ----------
    - stream_type: StreamInlet stream type; default is 'EEG'
    """

    def __init__(self,
                 stream_type: str = 'EEG',
                 max_timeout_seconds: float = 1.0):
        super(LslDataSource, self).__init__()

        print('Waiting for LSL EEG data stream...')
        self.stream_type = stream_type
        self.max_timeout_seconds = max_timeout_seconds
        streams = pylsl.resolve_stream('type', self.stream_type)
        self.inlet = pylsl.StreamInlet(streams[0])
        self.device_spec = device_from_metadata(self.inlet.info())

    def next(self):
        """Provide the next record."""
        sample, _ts = self.inlet.pull_sample(timeout=0.0)
        return sample

    def next_n(self, n: int, fast_forward=False):
        """Provides the next n records as a list

        Parameters:
        -----------
            n - number of records to retrieve from LSL
            fast_forward - if true, fast forwards to the latest data. This
                flag should be used when first connecting to LSL or resuming
                from a paused state. Otherwise the data returned will be the
                next n records from the last point consumed.
        """
        # Read data from the inlet. May blocks GUI interaction if n samples
        # are not yet available.
        samples, _ts = self.inlet.pull_chunk(timeout=self.max_timeout_seconds,
                                             max_samples=n)

        if fast_forward:
            tmp = samples
            print('Fast forwarding:')
            chomped_count = 0
            while len(tmp) == n:
                samples = tmp
                # A timeout of 0.0 does not block GUI interaction and only
                # gets samples immediately available.
                tmp, _ts = self.inlet.pull_chunk(timeout=0.0, max_samples=n)
                chomped_count += len(tmp)
            print(f'Chomped {chomped_count} records.')
            samples = samples[len(tmp):] + tmp

        if len(samples) < n and not fast_forward:
            # Fast forwarding generally occurs when the stream is first started
            # or when resuming after a pause, so there may not be a full n
            # samples available. However, if we are streaming normally and
            # less than n samples are returned, we assume the stream has
            # terminated.
            raise StopIteration
        return samples
