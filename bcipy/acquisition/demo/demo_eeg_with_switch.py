"""Sample script to demonstrate usage of LSL client and server."""
import subprocess
import time

from bcipy.config import BCIPY_ROOT
import logging
from bcipy.acquisition.datastream.mock.switch import switch_device
from bcipy.acquisition.devices import preconfigured_device
from bcipy.acquisition import LslAcquisitionClient, LslDataServer, await_start
from bcipy.helpers.system_utils import log_to_stdout

log = logging.getLogger(__name__)


def start_switch():
    """Start the demo switch"""
    return subprocess.Popen(f'python {BCIPY_ROOT}/acquisition/datastream/mock/switch.py',
                            shell=True)


def main(debug: bool = False):
    # pylint: disable=too-many-locals
    """Creates a sample lsl client that reads data from a sample LSL server
    (see demo/server.py).

    The client/server can be stopped with a Keyboard Interrupt (Ctl-C)."""

    if debug:
        log_to_stdout()

    eeg_device = preconfigured_device('DSI-24')
    eeg_server = LslDataServer(device_spec=eeg_device, log=log)

    eeg_client = LslAcquisitionClient(device_spec=eeg_device,
                                      save_directory='.',
                                      logger=log)
    switch_client = LslAcquisitionClient(max_buffer_len=1024,
                                         device_spec=switch_device(),
                                         save_directory='.',
                                         logger=log)
    await_start(eeg_server)

    # Open the Demo Switch GUI.
    start_switch()
    # Wait for switch start
    print("Waiting for Switch")
    time.sleep(0.5)
    try:
        seconds = 5
        switch_client.start_acquisition()
        eeg_client.start_acquisition()
        print(f"\nCollecting data for {seconds}s...",
              "Click in the demo switch GUI to register a switch hit.",
              "Close the GUI when finished.\n")

        time.sleep(seconds)

        eeg_client.stop_acquisition()
        switch_client.stop_acquisition()
        eeg_server.stop()
        print("\nThe collected data has been written to the local directory")

    except KeyboardInterrupt:
        print("Keyboard Interrupt; stopping.")
        eeg_client.stop_acquisition()
        switch_client.stop_acquisition()
        eeg_server.stop()
        print("\nThe collected data has been written to the local directory")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args.debug)
