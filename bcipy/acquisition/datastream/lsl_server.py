
import logging
import random
import time
from bcipy.acquisition.util import StoppableThread
from pylsl import StreamInfo, StreamOutlet
from queue import Queue, Empty
from bcipy.acquisition.datastream.producer import Producer

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)


class LslDataServer(StoppableThread):
    """Data server that streams EEG data over a LabStreamingLayer StreamOutlet
    using pylsl. See https://github.com/sccn/labstreaminglayer/wiki.

    In parameters.json, if the fake_data parameter is set to true and the
    device is set to LSL, this server will be used to mock data. Alternatively,
    fake_data can be set to false and this module can be run standalone in its
    own python instance.

    Parameters
    ----------
        params : dict(channels: list(str), hz(int))
            parameters used to configure the server. Should have at least
            a list of channels and the sample frequency.
        generator : Object (see generator.py for options)
            used to generate the data to be served. Ex. random_data.
        include_meta: bool, optional
            if True, writes metadata to the outlet stream.
        add_markers: bool, optional
            if True, creates a the marker channel and streams data to this
            channel at a fixed frequency.
    """

    def __init__(self, params, generator, include_meta=True,
                 add_markers=False, name='TestStream'):
        super(LslDataServer, self).__init__()

        self.channels = params['channels']
        self.hz = int(params['hz'])
        stream_name = params.get('name', name)

        params['channel_count'] = len(self.channels)
        self.channel_count = len(self.channels)
        self.generator = generator

        logging.debug("Starting server with params: " + str(params))
        info = StreamInfo(stream_name,
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

        self.add_markers = add_markers
        if add_markers:
            # Marker stream
            # lsl::stream_info marker_info("gUSBamp-"+deviceNumber+"Markers","Markers",1,0,lsl::cf_string,"gUSBamp_" + boost::lexical_cast<std::string>(deviceNumber) + "_" + boost::lexical_cast<std::string>(serialNumber) + "_markers");
            logging.debug("Creating marker stream")
            markers_info = StreamInfo("TestStream Markers",
                                      "Markers", 1, 0, 'string', "uid12345_markers")
            self.markers_outlet = StreamOutlet(markers_info)
        self.started = False

    def stop(self):
        """Stop the thread and cleanup resources."""

        logging.debug("[*] Stopping data server")
        super(LslDataServer, self).stop()

        # Allows pylsl to cleanup; The outlet will no longer be discoverable
        # after destruction and all connected inlets will stop delivering data.
        del self.outlet
        self.outlet = None

        if self.add_markers:
            del self.markers_outlet
            self.markers_outlet = None

    def run(self):
        """Main loop of the thread. Continuously streams data to the stream
        outlet at a rate consistent with the sample frequency. May also
        output markers at a different interval."""

        sample_counter = 0
        self.started = True

        q = Queue()
        with Producer(q, generator=self.generator, freq=1 / self.hz):
            while self.running():
                sample_counter += 1
                try:
                    sample = q.get(True, 2)
                    self.outlet.push_sample(sample)
                    if self.add_markers and sample_counter % 1000 == 0:
                        self.markers_outlet.push_sample(
                            [str(random.randint(1, 100))])
                except (Empty, AttributeError):
                    # outlet.push_sample(sample) may cause an error after
                    # the server has been stopped since the attribute is
                    # deleted in another thread.
                    break

        logging.debug("[*] No longer pushing data")


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

    from bcipy.acquisition.datastream.generator import file_data, random_data

    default_channels = ['ch' + str(i + 1) for i in range(16)]

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', default=None,
                        help="file containing data to be streamed; "
                        "if missing, random data will be served.")
    parser.add_argument('-c', '--channels',
                        default=','.join(default_channels),
                        help='comma-delimited list')
    parser.add_argument('-s', '--sample_rate', default='256',
                        help='sample rate in hz')

    parser.add_argument('-m', '--markers', action="store_true", default=False)
    parser.add_argument('-n', '--name', default='LSL')
    args = parser.parse_args()

    params = {'channels': args.channels.split(','),
              'hz': int(args.sample_rate)}

    # Generate data from the file if provided, otherwise random data.
    generator = file_data(filename=args.filename) if args.filename \
        else random_data(channel_count=len(params['channels']))

    markers = True if args.markers else False
    try:
        server = LslDataServer(params=params, generator=generator,
                               add_markers=markers, name=args.name)

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
