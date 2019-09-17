"""Demo the viewer used in conjunction with data acquisition."""


def main():
    """Creates a sample client that reads data from a sample TCP server
    (see demo/server.py). Data is written to a buffer.db sqlite3 database
    and streamed through a GUI. These files are written in whichever directory
    the script was run.

    The client/server can be stopped with a Keyboard Interrupt (Ctl-C)."""

    import time

    from bcipy.acquisition.datastream import generator
    from bcipy.acquisition.protocols import registry
    from bcipy.acquisition.client import DataAcquisitionClient
    from bcipy.acquisition.datastream.server import DataServer
    from bcipy.gui.viewer.processor.viewer_processor import ViewerProcessor

    host = '127.0.0.1'
    port = 9000
    # The Protocol is for mocking data.
    protocol = registry.default_protocol('DSI')
    server = DataServer(protocol=protocol,
                        generator=generator.random_data,
                        gen_params={'channel_count': len(protocol.channels)},
                        host=host, port=port)

    # Device is for reading data.
    # pylint: disable=invalid-name
    Device = registry.find_device('DSI')
    dsi_device = Device(connection_params={'host': host, 'port': port})
    client = DataAcquisitionClient(
        device=dsi_device,
        processor=ViewerProcessor())

    try:
        server.start()
        client.start_acquisition()
        seconds = 10
        print(
            f"\nCollecting data for {seconds}s... (Interrupt [Ctl-C] to stop)\n")

        t0 = time.time()
        elapsed = 0
        while elapsed < seconds:
            time.sleep(0.1)
            elapsed = (time.time()) - t0
        client.stop_acquisition()
        client.cleanup()
        print("Number of samples: {0}".format(client.get_data_len()))
        server.stop()

    except KeyboardInterrupt:
        print("Keyboard Interrupt; stopping.")
        client.stop_acquisition()
        client.cleanup()
        print("Number of samples: {0}".format(client.get_data_len()))
        server.stop()


if __name__ == '__main__':
    main()
