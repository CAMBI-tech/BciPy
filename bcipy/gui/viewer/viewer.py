"""
An example of how to use wx or wxagg in an application with the new
toolbar - comment out the setA_toolbar line for no toolbar
"""

from numpy import arange, sin, pi
import matplotlib

# uncomment the following to use wx rather than wxagg
# matplotlib.use('WX')
#from matplotlib.backends.backend_wx import FigureCanvasWx as FigureCanvas

# comment out the following to use wx rather than wxagg
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure

import wx
import wx.lib.mixins.inspection as WIT

import csv


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


class CanvasFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, -1,
                          'Demo: EEG Viewer', size=(550, 350))

        self.refresh_rate = 500  # ms
        # TODO: get hz from header or data source
        self.samples_per_second = 600.0
        # TODO: calculate from refresh rate and samples_per_second
        self.records_per_refresh = 300

        self.data_gen = file_data('raw_data.csv', n=self.records_per_refresh)
        self.header = next(self.data_gen)
        self.data_indices = [i for i in range(len(self.header))
                             if 'TRG' not in self.header[i] and 'timestamp' not in self.header[i]]

        self.figure = Figure()
        self.axis = self.figure.add_subplot(111)
        # key for each channel
        self.plot_data = {}

        self.canvas = FigureCanvas(self, -1, self.figure)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.EXPAND)
        self.SetSizer(self.sizer)
        self.Fit()
        self.init_data()

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update_data, self.timer)
        self.timer.Start(self.refresh_rate)

    def init_data(self):
        """Initialize the data."""
        rows = next(self.data_gen)
        timestamps = [float(r[0]) for r in rows]

        # plot each channel
        for ch in self.data_indices:
            data = [float(r[ch]) for r in rows]
            ch_name = self.header[ch]
            self.plot_data[ch_name] = self.axis.plot(
                timestamps, data, linewidth=0.8, label=ch_name)

        box = self.axis.get_position()
        self.axis.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        self.axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def update_data(self, evt):
        """Called by the timer on refresh."""
        try:
            rows = next(self.data_gen)
        except StopIteration as err:
            self.timer.Stop()
            return

        timestamps = [float(r[0]) for r in rows]

        # TODO: more efficient method of splitting out channels
        # plot each channel
        for ch in self.data_indices:
            data = [float(r[ch]) for r in rows]
            ch_name = self.header[ch]
            self.plot_data[ch_name][0].set_xdata(timestamps)
            self.plot_data[ch_name][0].set_ydata(data)

        # TODO: should y-axis be updated?
        self.axis.set_xbound(lower=timestamps[0], upper=timestamps[-1])
        self.canvas.draw()

# alternatively you could use
# class App(wx.App):


class App(WIT.InspectableApp):

    def OnInit(self):
        'Create the main window and insert the custom frame'
        self.Init()
        frame = CanvasFrame()
        frame.Show(True)

        return True


def main():
    """Run the viewer gui"""
    app = App(0)
    app.MainLoop()


if __name__ == "main":
    main()
