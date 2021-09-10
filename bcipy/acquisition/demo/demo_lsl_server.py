"""Demo a server with a custom generator"""
import argparse
import logging
import time

from bcipy.acquisition.datastream.lsl_server import LslDataServer
from bcipy.acquisition.devices import preconfigured_device

log = logging.getLogger(__name__)


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
        server = LslDataServer(device_spec=device_spec)

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
