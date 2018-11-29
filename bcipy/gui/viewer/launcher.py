"""Functionality for launching a new viewer in a separate Process/window."""
import wx
from bcipy.acquisition.util import StoppableProcess
import time


class Launcher(StoppableProcess):
    """Starts up the wx GUI in its own process. Any args required by the
    frame should be provided as keyword args.

    Parameters:
    -----------
        frame - wx Frame constructor that creates the viewer GUI. This should
            take two parameters, a data queue and device_info.       
    """

    def __init__(self, frame, **kwargs):
        super(Launcher, self).__init__()
        self.frame = frame
        self.args = kwargs

    def run(self):
        app = wx.App(False)
        frame = self.frame(**self.args)
        frame.Show(True)
        app.MainLoop()
