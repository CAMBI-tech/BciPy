"""
An example of how to use wx or wxagg in an application with the new
toolbar - comment out the setA_toolbar line for no toolbar
"""
import numpy as np
from numpy import arange, sin, pi
import matplotlib
# uncomment the following to use wx rather than wxagg
# matplotlib.use('WX')
# from matplotlib.backends.backend_wx import FigureCanvasWx as FigureCanvas

# comment out the following to use wx rather than wxagg
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.pyplot import NullLocator, NullFormatter

import wx
import wx.lib.mixins.inspection as WIT
from wx import BoxSizer, VERTICAL, LEFT, TOP, BOTTOM, EXPAND, EVT_TIMER, Timer, Frame
import csv

from bcipy.gui.viewer.ring_buffer import RingBuffer


def downsample(data, factor=2):
    """Decrease the sample rate of a sequence by a given factor."""
    # selector_pattern = [1] + ((factor-1) * [0])
    # return it.compress(data[offset:], it.cycle(selector_pattern))
    return np.array(data)[::factor]


def file_data(filename, n=100):
    """Read data from a raw_data.csv file"""
    with open(filename) as csvfile:
        # skip headers
        next(csvfile)
        next(csvfile)
        # reader = csv.DictReader(f)
        reader = csv.reader(csvfile)
        header = next(reader)
        yield header

        while True:
            rows = []
            for _ in range(n):
                rows.append(next(reader))
            yield rows


class CanvasFrame(Frame):
    def __init__(self, data_file='raw_data.csv', seconds=2, downsample_factor=2):
        Frame.__init__(self, None, -1,
                       'Demo: EEG Viewer', size=(800, 550))

        self.refresh_rate = 500  # ms
        # TODO: get hz from header or data source
        self.samples_per_second = 600.0
        self.records_per_refresh = int(
            (self.refresh_rate / 1000) * self.samples_per_second)

        self.data_gen = file_data(data_file, n=self.records_per_refresh)
        self.header = next(self.data_gen)
        self.data_indices = [i for i in range(len(self.header))
                             if 'TRG' not in self.header[i] and 'timestamp' not in self.header[i]]
        self.downsample_factor = downsample_factor
        buf_size = int((self.samples_per_second * seconds) /
                       self.downsample_factor)
        self.buffer = RingBuffer(buf_size, pre_allocated=True)

        # figure size is in inches.
        # https://stackoverflow.com/questions/332289/how-do-you-change-the-size-of-figures-drawn-with-matplotlib
        # figsize=(1200 / 80.0, 400 / 80.0)
        self.figure = Figure(figsize=(15, 10), dpi=80, tight_layout=True)
        self.axes = self.figure.subplots(
            len(self.data_indices), 1, sharex=True)

        self.subplots = {}
        for i, ch in enumerate(self.data_indices):
            ch_name = self.header[ch]
            self.axes[i].set_frame_on(False)
            self.axes[i].set_ylabel(ch_name, rotation=0, labelpad=15)
            # if i == len(self.data_indices) - 1:
            #     self.axes[i].set_xlabel("Sample")
            self.axes[i].yaxis.set_major_locator(NullLocator())
            self.axes[i].xaxis.set_major_formatter(NullFormatter())
            # x-axis label
            self.axes[i].yaxis.set_major_formatter(NullFormatter())
            # self.axes[i].xaxis.set_major_locator(NullLocator())

            # TODO: different color for frame.
            # self.axes[i].patch
            self.axes[i].grid()

        self.canvas = FigureCanvas(self, -1, self.figure)
        self.timer = Timer(self)
        self.Bind(EVT_TIMER, self.update_data, self.timer)

        self.CreateStatusBar()

        # Toolbar
        self.toolbar = wx.BoxSizer(wx.HORIZONTAL)
        self.start_stop_btn = wx.Button(self, -1, "Start")
        self.started = True

        self.Bind(wx.EVT_BUTTON, self.toggle_stream, self.start_stop_btn)
        self.toolbar.Add(self.start_stop_btn, 1, wx.ALIGN_CENTER, 0)

        self.sizer = BoxSizer(VERTICAL)
        self.sizer.Add(self.canvas, 1, LEFT | TOP | EXPAND)
        self.sizer.Add(self.toolbar, 0, wx.ALIGN_BOTTOM | wx. ALIGN_CENTER)
        self.SetSizer(self.sizer)
        self.SetAutoLayout(1)
        self.Fit()
        self.init_data()

        self.start()

    def start(self):
        """Start streaming data in the viewer."""
        self.timer.Start(self.refresh_rate)
        self.start_stop_btn.SetLabel("Pause")

    def stop(self):
        """Stop/Pause the viewer."""
        self.timer.Stop()
        self.start_stop_btn.SetLabel("Start")

    def toggle_stream(self, event):
        """Toggle data streaming"""
        if self.started:
            self.stop()
        else:
            self.start()
        self.started = not self.started

    def update_buffer(self):
        """Update the buffer with latest data and return the data"""
        try:
            for row in downsample(next(self.data_gen), self.downsample_factor):
                self.buffer.append(row)
        except StopIteration:
            self.stop()
        return self.buffer.get()

    def data_for_channel(self, ch, rows):
        """Extract the data for a given channel"""
        return [0.0 if r is None else float(r[ch]) for r in rows]

    def init_data(self):
        """Initialize the data."""
        rows = self.update_buffer()

        # plot each channel
        for i, ch in enumerate(self.data_indices):
            data = self.data_for_channel(ch, rows)
            self.axes[i].plot(data, linewidth=0.8)

    def update_data(self, evt):
        """Called by the timer on refresh."""
        rows = self.update_buffer()

        # TODO: more efficient method of splitting out channels
        # plot each channel
        for i, ch in enumerate(self.data_indices):
            data = self.data_for_channel(ch, rows)
            self.axes[i].lines[0].set_ydata(data)
            # TODO: should y-axis be updated?
            self.axes[i].set_ybound(lower=min(data), upper=max(data))

        self.canvas.draw()


class App(WIT.InspectableApp):

    def OnInit(self):
        'Create the main window and insert the custom frame'
        self.Init()
        frame = CanvasFrame()
        frame.Show(True)

        return True


def main(data_file, seconds, downsample):
    """Run the viewer gui"""
    # app = App(False)
    app = wx.App(False)
    frame = CanvasFrame(data_file, seconds, downsample)
    frame.Show(True)
    app.MainLoop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--file', help='path to the data file', default='raw_data.csv')
    parser.add_argument('-s', '--seconds', help='seconds to display', default=2, type=int)
    parser.add_argument('-d', '--downsample', help='downsample factor', default=2, type=int)
    args = parser.parse_args()
    main(args.file, args.seconds, args.downsample)
