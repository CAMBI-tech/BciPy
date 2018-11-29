from bcipy.acquisition.processor import Processor
from bcipy.gui.viewer.pubsub_viewer import Viewer
from multiprocessing import JoinableQueue


class ViewerProcessor(Processor):
    """Processor that displays the streaming data in a GUI."""

    def __init__(self):
        super(ViewerProcessor, self).__init__()
        self.data_queue = JoinableQueue()
        self.viewer = None

    # @override ; context manager
    def __enter__(self):
        self._check_device_info()

        self.viewer = Viewer(self.data_queue, self._device_info)
        self.viewer.start()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.viewer.stop()

    def process(self, record, timestamp=None):
        self.data_queue.put(record)
