"""Sample program to demonstrate creating a data server."""


def main():
    """Creates a new TCP server that serves up random EEG-like data in the
    DSI format. The server can be stopped with a Keyboard Interrupt (Ctl-C)"""

    import time
    import sys

    # Allow the script to be run from the bci root, acquisition dir, or
    # demo dir.
    sys.path.append('.')
    sys.path.append('..')
    sys.path.append('../..')

    from bcipy.acquisition.datastream.generator import random_data_generator, generator_factory
    from bcipy.acquisition.protocols import registry
    from bcipy.acquisition.datastream.tcp_server import TcpDataServer

    # Find the DSI protocol by name.
    protocol = registry.default_protocol('DSI')

    try:
        server = TcpDataServer(
            protocol=protocol,
            generator=generator_factory(random_data_generator, channel_count=len(protocol.channels)),
            host='127.0.0.1',
            port=9000)
        server.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        server.stop()


if __name__ == '__main__':
    main()
