import csv
from bcipy.acquisition.util import StoppableThread


class FileStreamer(StoppableThread):
    """Process that continuously reads data from the data source and publishes
    it at the frequency specified in the raw_data.


    Parameters
    ----------
        data_file: str 
            raw data file from which to read
        data_queue : Queue
            Data will be written to the queue as it is acquired.
        msg_queue : Queue
            Used to communicate messages to the main thread.
    """

    def __init__(self, data_file, data_queue):
        super(FileStreamer, self).__init__()
        self.data_file = data_file
        self.data_queue = data_queue

    def run(self):
        print("Starting streamer")
        import time
        with open(self.data_file) as csvfile:
            # read metadata
            _name_row = next(csvfile)
            fs = float(next(csvfile).strip().split(",")[-1])

            reader = csv.reader(csvfile)
            _channels = next(reader)

            print("Publishing data")
            # publish data
            for data in reader:
                if not self.running():
                    break
                self.data_queue.put(data)
                time.sleep(1/fs)
            print("Stopping file streamer")
