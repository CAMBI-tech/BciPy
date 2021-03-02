"""Demo a server with a custom generator"""
import time
import argparse

from bcipy.acquisition.datastream.generator import file_data_generator
from bcipy.acquisition.devices import preconfigured_device
from bcipy.acquisition.datastream.lsl_server import LslDataServer
from bcipy.acquisition.devices import DeviceSpec
from bcipy.signal.generator.generator import gen_random_data
import logging
log = logging.getLogger(__name__)


def custom_generator(device_spec: DeviceSpec, low=-1000, high=1000):
    """Generates sequential data for the first channel and random data for the
    rest. The low and high parameters set the bounds for the random data.    
    """
    i = 0
    while True:
        i += 1
        sensor_data = gen_random_data(low, high, device_spec.channel_count - 1)
        yield [i] + sensor_data


def main():
    """Initialize and start the server."""

    parser = argparse.ArgumentParser()

    parser.add_argument('-n',
                        '--name',
                        default='LSL',
                        help='Name of the device spec to mock.')
    args = parser.parse_args()
    device_spec = preconfigured_device(args.name)
    try:
        server = LslDataServer(device_spec=device_spec,
                               generator=custom_generator(device_spec))

        log.debug("New server created")
        server.start()
        log.debug("Server started")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        server.stop()


if __name__ == '__main__':
    main()