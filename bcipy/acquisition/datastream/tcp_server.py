"""TCP Server to stream mock EEG data."""

import logging
from queue import Queue, Empty
import select
import socket
import time

from bcipy.acquisition.datastream.producer import Producer
from bcipy.acquisition.datastream.generator import random_data_generator
from bcipy.acquisition.util import StoppableThread
from bcipy.helpers.raw_data import settings

log = logging.getLogger(__name__)


class TcpDataServer(StoppableThread):
    """Streams sample EEG-like data via TCP.

    Parameters
    ----------
        protocol : DeviceProtocol
            protocol-specific behavior including fs, init_messages, and encoder
        generator : function, optional
            generator function; used to generate the data to be served. Default
            behavior is to use a random_data_generator.
        host : str, optional
        port : int, optional
    """

    # pylint: disable=too-many-arguments
    def __init__(self, protocol=None, generator=None, host='127.0.0.1', port=9999):
        super(TcpDataServer, self).__init__(name="TcpDataServer")
        self.protocol = protocol
        self.generator = generator or random_data_generator

        self.host = host
        self.port = port

        self.started = False

        # Putting socket.bind in constructor allows us to handle any errors
        # (such as Address already in use) in the calling context.
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,
                                      1)
        self.server_socket.bind((self.host, self.port))

    def run(self):
        """Listen for client connection and initiate handler."""

        self.server_socket.listen(1)

        self.started = True
        log.debug("[*] Accepting connections on %s:%d", self.host, self.port)

        while self.running():

            # Sockets from which we expect to read
            inputs = [self.server_socket]
            outputs = []
            errors = []

            # Polls for readable state
            readable, _w, _e = select.select(inputs, outputs, errors, 0.05)

            if readable:
                client, addr = self.server_socket.accept()
                log.debug("[*] Accepted connection from: %s:%d", addr[0],
                          addr[1])
                self._handle_client(client)
        try:
            self.server_socket.shutdown(2)
            self.server_socket.close()
        # pylint: disable=broad-except
        except Exception:
            pass
        self.server_socket = None
        log.debug("[*] No longer accepting connections")

    def stop(self):
        log.debug("[*] Stopping data server")
        super(TcpDataServer, self).stop()

    def init_messages(self):
        """Messages sent to a client_socket on initialization before sending data."""
        if self.protocol:
            return self.protocol.init_messages()
        return []

    def make_generator(self):
        """Constructs a new data generator given the configured generator function."""
        assert self.protocol, "Device-specific information must be set."
        return self.generator(channel_count=len(self.protocol.channels),
                              encoder=self.protocol)

    def sample_rate(self) -> float:
        """Frequency at which data is generated, in hz"""
        assert self.protocol, "Device-specific information must be set."
        return self.protocol.sample_rate

    def _handle_client(self, client_socket):
        """This currently only handles a single client and blocks on that call.

        If multiple clients need to connect, the server needs to track the
        clients and send the data to each one (each of which would have a queue
        of data to be sent)
        """
        wfile = client_socket.makefile(mode='wb')

        # Send initialization data according to the protocol
        for msg in self.init_messages():
            wfile.write(msg)

        # This needs to be handled differently if we decide to allow
        # multiple clients. Producer needs to be managed by the class,
        # and the same message should be sent to each client.
        data_queue = Queue()

        # Construct a new generator each time to get consistent results.
        with Producer(data_queue,
                      generator=self.make_generator(),
                      freq=1 / self.sample_rate()):
            while self.running():
                try:
                    # block if necessary, for up to 5 seconds
                    item = data_queue.get(True, 5)
                except Empty:
                    client_socket.close()
                    break
                try:
                    wfile.write(item)
                except IOError:
                    break
        client_socket.close()
        log.debug("[*] Client disconnected")


def start_socket_server(protocol, host, port, retries=2):
    """Starts a TcpDataServer given the provided port and host information. If
    the port is not available, will automatically try a different port up to
    the given number of times. Returns the server along with the port.

    Parameters
    ----------
        protocol : Protocol for how to generate data.
        host : str ; socket host (ex. '127.0.0.1').
        port : int.
        retries : int; number of times to attempt another port if provided
            port is busy.
    Returns
    -------
        (server, port)
    """
    try:
        dataserver = TcpDataServer(protocol=protocol, host=host, port=port)

    except IOError as error:
        if retries > 0:
            # try a different port when 'Address already in use'.
            port = port + 1
            log.debug("Address in use: trying port %d", port)
            return start_socket_server(protocol, host, port, retries - 1)
        raise error

    await_start(dataserver)
    return dataserver, port


def await_start(dataserver, max_wait=2):
    """Blocks until server is started. Raises if max_wait is exceeded before
    server is started."""

    dataserver.start()
    wait = 0
    wait_interval = 0.01
    while not dataserver.started:
        time.sleep(wait_interval)
        wait += wait_interval
        if wait >= max_wait:
            dataserver.stop()
            raise Exception("Server couldn't start up in time.")


def main():
    """Initialize and run the server."""
    import argparse

    from bcipy.acquisition.datastream.generator import file_data_generator, random_data_generator, generator_with_args
    from bcipy.acquisition.protocols.registry import find_protocol
    from bcipy.acquisition.connection_method import ConnectionMethod
    from bcipy.acquisition.devices import preconfigured_device

    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--host', default='127.0.0.1')
    parser.add_argument('-p', '--port', type=int, default=8844)
    parser.add_argument('-f',
                        '--filename',
                        default=None,
                        help="file containing data to be streamed; "
                        "if missing, random data will be served.")
    parser.add_argument('-n', '--name', default='DSI', help='Name of the device spec to mock.')
    args = parser.parse_args()

    if args.filename:
        daq_type, sample_rate, channels = settings(args.filename)
        device_spec = preconfigured_device(daq_type)
        device_spec.sample_rate = sample_rate
        device_spec.channels = channels

        generator = generator_with_args(file_data_generator,
                                        filename=args.filename)
    else:
        device_spec = preconfigured_device(args.name)
        generator = random_data_generator

    protocol = find_protocol(device_spec, ConnectionMethod.TCP)
    try:
        server = TcpDataServer(protocol=protocol,
                               generator=generator,
                               host=args.host,
                               port=args.port)
        log.debug("New server created")
        server.start()
        log.debug("Server started")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        server.stop()


if __name__ == '__main__':
    # Run this as: python -m acquisition.datastream.server
    main()
