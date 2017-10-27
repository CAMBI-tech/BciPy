"""TCP Server to stream mock EEG data."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import errno
import logging
import Queue
import select
import socket
import threading

from datastream.producer import Producer

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)


class DataServer(threading.Thread):
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

    def __init__(self, protocol, generator, gen_params={}, host='0.0.0.0',
                 port=9999):
        super(DataServer, self).__init__()
        self.protocol = protocol
        self.generator = generator
        self.gen_params = gen_params
        self.gen_params['encoder'] = protocol.encoder

        self.host = host
        self.port = port
        self.running = True

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))

        self.server.listen(2)
        logging.debug("[*] Listening on %s:%d" % (self.host, self.port))

    def run(self):
        self.serverthread = threading.Thread(target=self._accept_clients,
                                             name="TCP Server")
        self.serverthread.daemon = True
        self.serverthread.start()

    def _accept_clients(self):
        """Listen for client connection and initiate handler."""

        logging.debug("Accepting connections")
        while self.running:

            inputs = [self.server]  # Sockets from which we expect to read
            outputs = []
            errors = []

            # Polls for readable state
            readable, w, e = select.select(inputs, outputs, errors, 0.05)

            if readable:
                client, addr = self.server.accept()
                logging.debug("[*] Accepted connection from: %s:%d" %
                              (addr[0], addr[1]))
                self._handle_client(client)

    def stop(self):
        logging.debug("Stopping server")
        self.running = False
        self.serverthread.join()

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
        q = Queue.Queue()

        # Construct a new generator each time to get consistent results.
        generator = self.generator(**self.gen_params)
        with Producer(q, generator=generator, freq=1 / self.protocol.fs):
            while self.running:
                try:
                    # block if necessary, for up to 5 seconds
                    item = q.get(True, 5)
                except Queue.Empty:
                    client_socket.close()
                    break
                try:
                    wfile.write(item)

                except IOError as e:
                    if e.errno == errno.EPIPE:
                        break
        logging.debug("Client disconnected")


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

    from datastream.generator import file_data, random_data
    from protocols.registry import protocol_with, default_protocol

    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--host', default='0.0.0.0')
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
