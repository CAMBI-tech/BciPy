"""Displays the data_viewer GUI on initialization. ViewerProcessor subclasses
the data acquisition Processor so it can be passed in to the
DataAcquisitionClient constructor."""
import subprocess
import time

from bcipy.acquisition.processor import Processor
from bcipy.config import BCIPY_ROOT


class ViewerProcessor(Processor):
    """Processor that displays the streaming data in a GUI.

    Parameters:
    ----------
        display_screen - monitor on which to display the viewer. The default
            is 1, which is the secondary monitor. A value of 0 is the primary.
    """

    def __init__(self, display_screen: int = 1):
        super(ViewerProcessor, self).__init__()
        self.display_screen = display_screen
        self.viewer = f'{BCIPY_ROOT}/gui/viewer/data_viewer.py'
        self.started = False

    # @override ; context manager
    def __enter__(self):
        self._check_device_info()

        cmd = f'python {self.viewer} -m {self.display_screen}'
        # On mac/linux, we can close the viewer in the exit method. However,
        # this doesn't work on Windows. For now, leave it to the user to
        # close.

        # https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true
        # self.subproc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
        #                                 shell=True, preexec_fn=os.setsid)
        self.subproc = subprocess.Popen(cmd, shell=True)

        # hack: wait for window to open, so it doesn't error out when the main
        # window is open fullscreen.
        time.sleep(2)
        self.started = True
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        # On Mac this closes the viewer window; see above comment.
        # os.killpg(os.getpgid(self.subproc.pid), signal.SIGTERM)
        self.started = False

    def process(self, record, timestamp=None):
        pass
