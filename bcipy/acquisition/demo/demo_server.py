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

    import bcipy.acquisition.datastream.generator as generator
    import bcipy.acquisition.protocols.registry as registry
    from bcipy.acquisition.datastream.server import DataServer

    # Find the DSI protocol by name.
    protocol = registry.default_protocol('DSI')

    try:
        server = DataServer(protocol=protocol,
                            generator=generator.random_data,
                            gen_params={'channel_count': len(
                                protocol.channels)},
                            host='127.0.0.1', port=9000)
        server.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        server.stop()


if __name__ == '__main__':
    main()
