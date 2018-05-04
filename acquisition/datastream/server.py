"""TCP Server to stream mock EEG data."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import errno
import logging
from queue import Queue, Empty
import select
import socket
import time
import threading


from acquisition.datastream.producer import Producer
from acquisition.util import StoppableThread

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)


class DataServer(StoppableThread):
    """Streams sample EEG-like data via TCP.

    Parameters
    ----------
        protocol : Protocol
            protocol-specific behavior including fs, init_messages, and encoder
        generator : function
            generator function; used to generate the data to be served. A new
            generator will be created for every client connection.
        gen_params : dict, optional
            parameters for the generator function
        host : str, optional
        port : int, optional
    """

    def __init__(self, protocol, generator, gen_params={}, host='127.0.0.1',
                 port=9999):
        super(DataServer, self).__init__(name="DataServer")
        self.protocol = protocol
        self.generator = generator
        self.gen_params = gen_params
        self.gen_params['encoder'] = protocol.encoder

        self.host = host
        self.port = port

        self.started = False

        # Putting socket.bind in constructor allows us to handle any errors
        # (such as Address already in use) in the calling context.
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))

    def run(self):
        """Listen for client connection and initiate handler."""

        self.server_socket.listen(1)

        self.started = True
        logging.debug("[*] Accepting connections on %s:%d" %
                      (self.host, self.port))

        while self.running():

            # Sockets from which we expect to read
            inputs = [self.server_socket]
            outputs = []
            errors = []

            # Polls for readable state
            readable, w, e = select.select(inputs, outputs, errors, 0.05)

            if readable:
                client, addr = self.server_socket.accept()
                logging.debug("[*] Accepted connection from: %s:%d" %
                              (addr[0], addr[1]))
                self._handle_client(client)
        try:
            self.server_socket.shutdown(2)
            self.server_socket.close()
        except Exception as e:
            pass
        self.server_socket = None
        logging.debug("[*] No longer accepting connections")

    def stop(self):
        logging.debug("[*] Stopping data server")
        super(DataServer, self).stop()

    def _handle_client(self, client_socket):
        """This currently only handles a single client and blocks on that call.

        If multiple clients need to connect, the server needs to track the
        clients and send the data to each one (each of which would have a queue
        of data to be sent)
        """
        wfile = client_socket.makefile(mode='wb')

        # Send initialization data according to the protocol
        for msg in self.protocol.init_messages:
            wfile.write(msg)

        # This needs to be handled differently if we decide to allow
        # multiple clients. Producer needs to be managed by the class,
        # and the same message should be sent to each client.
        q = Queue()

        # Construct a new generator each time to get consistent results.
        generator = self.generator(**self.gen_params)
        with Producer(q, generator=generator, freq=1 / self.protocol.fs):
            while self.running():
                try:
                    # block if necessary, for up to 5 seconds
                    item = q.get(True, 5)
                except Empty:
                    client_socket.close()
                    break
                try:
                    wfile.write(item)
                except IOError as e:
                    break
        client_socket.close()
        logging.debug("[*] Client disconnected")


def start_socket_server(protocol, host, port, retries=2):
    """Starts a DataServer given the provided port and host information. If
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
    from acquisition.datastream.generator import random_data
    try:
        dataserver = DataServer(protocol=protocol,
                                generator=random_data,
                                gen_params={'channel_count': len(
                                    protocol.channels)},
                                host=host,
                                port=port)

    except IOError as e:
        if retries > 0:
            # try a different port when 'Address already in use'.
            port = port + 1
            logging.debug("Address in use: trying port {}".format(port))
            return start_socket_server(protocol, host, port, retries - 1)
        else:
            raise e

    await_start(dataserver)
    return dataserver, port


def await_start(dataserver, max_wait=2):
    """Blocks until server is started. Raises if max_wait is exceeded before
    server is started."""
    import time

    dataserver.start()
    wait = 0
    wait_interval = 0.01
    while not dataserver.started:
        time.sleep(wait_interval)
        wait += wait_interval
        if wait >= max_wait:
            dataserver.stop()
            raise Exception("Server couldn't start up in time.")


def _settings(filename):
    """Read the daq settings from the given data file"""

    with open(filename, 'r') as f:
        daq_type = f.readline().strip().split(',')[1]
        fs = int(f.readline().strip().split(',')[1])
        channels = f.readline().strip().split(',')
        return (daq_type, fs, channels)


if __name__ == '__main__':
    # Run this as: python -m acquisition.datastream.server

    import time
    import argparse

    from acquisition.datastream.generator import file_data, random_data
    from acquisition.protocols.registry import protocol_with, default_protocol

    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--host', default='127.0.0.1')
    parser.add_argument('-p', '--port', type=int, default=8844)
    parser.add_argument('-f', '--filename', default=None,
                        help="file containing data to be streamed; "
                        "if missing, random data will be served.")
    args = parser.parse_args()

    if args.filename:
        daq_type, fs, channels = _settings(args.filename)
        protocol = protocol_with(daq_type, fs, channels)
        generator, params = (file_data, {'filename': args.filename})
    else:
        protocol = default_protocol('DSI')
        channel_count = len(protocol.channels)
        generator, params = (random_data, {'channel_count': channel_count})

    try:
        server = DataServer(protocol=protocol,
                            generator=generator,
                            gen_params=params,
                            host=args.host, port=args.port)
        logging.debug("New server created")
        server.start()
        logging.debug("Server started")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        server.stop()
