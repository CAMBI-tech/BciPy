
import logging
import time
from acquisition.util import StoppableThread
from pylsl import StreamInfo, StreamOutlet

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)


class LslDataServer(StoppableThread):
    """Data server that streams EEG data using pylsl."""

    def __init__(self, params, generator, include_meta=True):
        super(LslDataServer, self).__init__()

        self.channels = params['channels']
        self.hz = int(params['hz'])
        params['channel_count'] = len(self.channels)
        self.channel_count = len(self.channels)
        self.generator = generator

        logging.debug("Starting server with params: " + str(params))
        info = StreamInfo("TestStream",
                          "EEG", self.channel_count,
                          self.hz,
                          'float32',
                          "uid12345")

        if include_meta:
            meta_channels = info.desc().append_child('channels')
            for c in params['channels']:
                meta_channels.append_child('channel') \
                    .append_child_value('label', c) \
                    .append_child_value('unit', 'microvolts') \
                    .append_child_value('type', 'EEG')

        self.outlet = StreamOutlet(info)

    def stop(self):
        super(LslDataServer, self).stop()

        # Allows pylsl to cleanup; The outlet will no longer be discoverable
        # after destruction and all connected inlets will stop delivering data.
        del self.outlet
        self.outlet = None

    def next_sample(self):
        return next(self.generator)

    def run(self):
        while self.running():
            self.outlet.push_sample(self.next_sample())
            time.sleep(1 / self.hz)


def _settings(filename):
    """Read the daq settings from the given data file"""

    with open(filename, 'r') as f:
        daq_type = f.readline().strip().split(',')[1]
        fs = int(f.readline().strip().split(',')[1])
        channels = f.readline().strip().split(',')
        return (daq_type, fs, channels)


def main():
    import time
    import argparse

    from acquisition.datastream.generator import file_data, random_data

    default_channels = ['ch' + str(i + 1) for i in range(24)]
    default_channels.append("TRG")

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', default=None,
                        help="file containing data to be streamed; "
                        "if missing, random data will be served.")
    parser.add_argument('-c', '--channels',
                        default=','.join(default_channels),
                        help='comma-delimited list')
    parser.add_argument('-s', '--sample_rate', default='300',
                        help='sample rate in hz')
    args = parser.parse_args()

    params = {'channels': args.channels.split(','),
              'hz': int(args.sample_rate)}

    # Generate data from the file if provided, otherwise random data.
    generator = file_data(filename=args.filename) if args.filename \
        else random_data(channel_count=len(params['channels']))

    try:
        server = LslDataServer(params=params, generator=generator)

        logging.debug("New server created")
        server.start()
        logging.debug("Server started")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        server.stop()


if __name__ == '__main__':
    # Run this as: python -m acquisition.datastream.lsl_server
    main()
