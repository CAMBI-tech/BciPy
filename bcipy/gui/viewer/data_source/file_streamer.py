"""Streams file data for the viewer"""
import logging
import time

from bcipy.acquisition.util import StoppableThread
from bcipy.config import SESSION_LOG_FILENAME
from bcipy.core.raw_data import RawDataReader

log = logging.getLogger(SESSION_LOG_FILENAME)


class FileStreamer(StoppableThread):
    """Process that continuously reads data from a raw_data.csv file and
    publishes it to a Queue at the frequency specified in the metadata.

    Parameters
    ----------
        data_file: str
            raw data file from which to read
        data_queue : Queue
            Data will be written to the queue as it is acquired.
    """

    def __init__(self, data_file, data_queue):
        super(FileStreamer, self).__init__()
        self.data_file = data_file
        self.data_queue = data_queue

    def run(self):
        log.info("Starting raw_data file streamer")

        with RawDataReader(self.data_file, convert_data=True) as reader:
            fs = reader.sample_rate
            log.info(f"Publishing data at sample rate {fs} hz")

            # publish data
            for data in reader:
                if not self.running():
                    break
                self.data_queue.put(data)
                time.sleep(1 / fs)
            log.info("Stopping raw_data file streamer")
