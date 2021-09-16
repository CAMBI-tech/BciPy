"""Demo the viewer used in conjunction with data acquisition."""


def main():
    """Creates a sample client that reads data from a sample TCP server
    (see demo/server.py). Data is written to a buffer.db sqlite3 database
    and streamed through a GUI. These files are written in whichever directory
    the script was run.

    The client/server can be stopped with a Keyboard Interrupt (Ctl-C)."""

    from bcipy.acquisition.datastream.lsl_server import LslDataServer
    from bcipy.acquisition.devices import preconfigured_device
    from bcipy.gui.viewer import data_viewer

    device_spec = preconfigured_device('LSL')
    server = LslDataServer(device_spec=device_spec)

    try:
        server.start()
        data_viewer.main(data_file=None,
                         seconds=5,
                         refresh=500,
                         yscale=150,
                         display_screen=0)
        # Stop the server after the data_viewer GUI is closed
        server.stop()
    except KeyboardInterrupt:
        print("Keyboard Interrupt; stopping.")
        server.stop()


if __name__ == '__main__':
    main()
