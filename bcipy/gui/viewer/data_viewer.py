# pylint: disable=no-name-in-module,no-member,wrong-import-position,ungrouped-imports
"""
EEG viewer that uses a queue as a data source. Records are retrieved by a
wx Timer.
"""
import csv
from queue import Queue

import matplotlib
import numpy as np
import wx
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.pyplot import NullFormatter, NullLocator
import matplotlib.ticker as ticker

from bcipy.acquisition.device_info import DeviceInfo
from bcipy.acquisition.util import StoppableProcess
from bcipy.gui.viewer.data_source import QueueDataSource
from bcipy.gui.viewer.launcher import Launcher
from bcipy.gui.viewer.ring_buffer import RingBuffer
from bcipy.gui.viewer.filter import downsample_filter, sig_pro_filter


class EegFrame(wx.Frame):
    """GUI Frame in which data is plotted. Plots a subplot for every channel.
    Relies on a Timer to retrieve data at a specified interval. Data to be
    displayed is retrieved from a Queue.

    Parameters:
    -----------
        data_source - data source
        device_info - metadata about the data.
        seconds - how many seconds worth of data to display.
        downsample_factor - how much to compress the data. A factor of 1
            displays the raw data.
        refresh - time in milliseconds; how often to refresh the plots
    """

    def __init__(self, data_source, device_info: DeviceInfo,
                 seconds: int = 5, downsample_factor: int = 2,
                 refresh: int = 500,
                 y_scale=100):
        wx.Frame.__init__(self, None, -1,
                          'EEG Viewer', size=(800, 550))

        self.data_source = data_source

        self.refresh_rate = refresh
        self.samples_per_second = device_info.fs
        self.records_per_refresh = int(
            (self.refresh_rate / 1000) * self.samples_per_second)

        self.channels = device_info.channels
        self.removed_channels = ['TRG', 'timestamp']
        self.data_indices = self.init_data_indices()

        self.seconds = seconds
        self.downsample_factor = downsample_factor
        self.filter = downsample_filter(downsample_factor, device_info.fs)

        self.autoscale = True
        self.y_min = -y_scale
        self.y_max = y_scale

        self.buffer = self.init_buffer()

        # figure size is in inches.
        self.figure = Figure(figsize=(12, 9), dpi=80,
                             tight_layout={'pad': 0.0})
        self.axes = self.init_axes()

        self.canvas = FigureCanvas(self, -1, self.figure)

        self.CreateStatusBar()

        # Toolbar
        self.toolbar = wx.BoxSizer(wx.VERTICAL)

        self.start_stop_btn = wx.Button(self, -1, "Start")

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update_view, self.timer)
        self.Bind(wx.EVT_BUTTON, self.toggle_stream, self.start_stop_btn)

        # Filtered checkbox
        self.sigpro_checkbox = wx.CheckBox(self, label="Filtered")
        self.sigpro_checkbox.SetValue(False)
        self.Bind(wx.EVT_CHECKBOX, self.toggle_filtering_handler,
                  self.sigpro_checkbox)

        # Autoscale checkbox
        self.autoscale_checkbox = wx.CheckBox(self, label="Autoscale")
        self.autoscale_checkbox.SetValue(self.autoscale)
        self.Bind(wx.EVT_CHECKBOX, self.toggle_autoscale_handler,
                  self.autoscale_checkbox)

        # Number of seconds text box
        self.seconds_choices = [2, 5, 10]
        if not self.seconds in self.seconds_choices:
            self.seconds_choices.append(self.seconds)
            self.seconds_choices.sort()
        opts = [str(x) + " seconds" for x in self.seconds_choices]
        self.seconds_input = wx.Choice(self, choices=opts)
        cur_sec_selection = self.seconds_choices.index(self.seconds)
        self.seconds_input.SetSelection(cur_sec_selection)
        self.Bind(wx.EVT_CHOICE, self.seconds_handler, self.seconds_input)

        controls = wx.BoxSizer(wx.HORIZONTAL)
        controls.Add(self.start_stop_btn, 1, wx.ALIGN_CENTER, 0)
        controls.Add(self.sigpro_checkbox, 1, wx.ALIGN_CENTER, 0)
        controls.Add(self.autoscale_checkbox, 1, wx.ALIGN_CENTER, 0)
        # TODO: pull right; currently doesn't do that
        controls.Add(self.seconds_input, 0, wx.ALIGN_RIGHT, 0)

        self.toolbar.Add(controls, 1, wx.ALIGN_CENTER, 0)
        self.init_channel_buttons()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.EXPAND)
        sizer.Add(self.toolbar, 0, wx.ALIGN_BOTTOM | wx. ALIGN_CENTER)
        self.SetSizer(sizer)
        self.SetAutoLayout(1)
        self.Fit()

        self.init_data()
        self.started = False
        self.start()

    def init_data_indices(self):
        return [i for i in range(len(self.channels))
                if self.channels[i] not in self.removed_channels
                and 'TRG' not in self.channels[i]]

    def init_buffer(self):
        """Initialize the buffer"""
        buf_size = int(self.samples_per_second * self.seconds)
        empty_val = [0.0 for _x in self.channels]
        buf = RingBuffer(buf_size, pre_allocated=True, empty_value=empty_val)
        return buf

    def init_axes(self):
        """Sets configuration for axes"""
        axes = self.figure.subplots(len(self.data_indices), 1, sharex=True)
        for i, channel in enumerate(self.data_indices):
            ch_name = self.channels[channel]
            axes[i].set_frame_on(False)
            axes[i].set_ylabel(ch_name, rotation=0, labelpad=15)
            # x-axis shows seconds in 0.5 sec increments
            tick_names = np.arange(0, self.seconds, 0.5)
            ticks = [(self.samples_per_second * sec) / self.downsample_factor
                     for sec in tick_names]
            axes[i].xaxis.set_major_locator(ticker.FixedLocator(ticks))
            axes[i].xaxis.set_major_formatter(
                ticker.FixedFormatter(tick_names))
            # axes[i].yaxis.set_major_locator(NullLocator())
            # axes[i].yaxis.set_major_formatter(NullFormatter())
            axes[i].grid()
        return axes

    def reset_axes(self):
        """Clear the data in the gui."""
        self.figure.clear()
        self.axes = self.init_axes()

    def init_channel_buttons(self):
        """Add buttons for toggling the channels."""

        channel_box = wx.BoxSizer(wx.HORIZONTAL)
        for channel_index in self.data_indices:
            channel = self.channels[channel_index]
            chkbox = wx.CheckBox(self, label=channel, id=channel_index)
            chkbox.SetValue(channel not in self.removed_channels)

            self.Bind(wx.EVT_CHECKBOX, self.toggle_channel, chkbox)
            channel_box.Add(chkbox, 0, wx.ALIGN_CENTER, 0)

        self.toolbar.Add(channel_box, 1, wx.ALIGN_LEFT, 0)

    def toggle_channel(self, event):
        """Remove the provided channel from the display"""
        channel_index = event.GetEventObject().GetId()
        channel = self.channels[channel_index]

        previously_running = self.started
        if self.started:
            self.stop()

        if channel in self.removed_channels:
            self.removed_channels.remove(channel)
        else:
            self.removed_channels.append(channel)
        self.data_indices = self.init_data_indices()
        self.reset_axes()
        self.init_data()
        self.canvas.draw()
        if previously_running:
            self.start()

    def start(self):
        """Start streaming data in the viewer."""
        # update buffer with latest data on (re)start.
        self.update_buffer(fast_forward=True)
        self.timer.Start(self.refresh_rate)
        self.started = True
        self.start_stop_btn.SetLabel("Pause")

    def stop(self):
        """Stop/Pause the viewer."""
        self.timer.Stop()
        self.started = False
        self.start_stop_btn.SetLabel("Start")

    def toggle_stream(self, _event):
        """Toggle data streaming"""
        if self.started:
            self.stop()
        else:
            self.start()

    def toggle_filtering_handler(self, event):
        """Event handler for toggling data filtering."""
        self.with_refresh(self.toggle_filtering)

    def toggle_filtering(self):
        """Toggles data filtering."""
        if self.sigpro_checkbox.GetValue():
            self.filter = sig_pro_filter(
                self.downsample_factor, self.samples_per_second)
        else:
            self.filter = downsample_filter(
                self.downsample_factor, self.samples_per_second)

    def toggle_autoscale_handler(self, event):
        """Event handler for toggling autoscale"""
        self.with_refresh(self.toggle_autoscale)

    def toggle_autoscale(self):
        """Sets autoscale to checkbox value"""
        self.autoscale = self.autoscale_checkbox.GetValue()

    def seconds_handler(self, event):
        """Event handler for changing seconds"""
        self.with_refresh(self.update_seconds)

    def update_seconds(self):
        """Set the number of seconds worth of data to display from the
        pulldown list."""
        self.seconds = self.seconds_choices[self.seconds_input.GetSelection()]
        self.buffer = self.init_buffer()

    def with_refresh(self, fn):
        """Performs the given action and refreshes the display."""
        previously_running = self.started
        if self.started:
            self.stop()
        fn()
        # re-initialize
        self.reset_axes()
        self.init_data()
        if previously_running:
            self.start()

    def current_data(self):
        """Returns the data as an np array with a row of floats for each
        displayed channel."""

        # array of 'object'; TRG data may be strings
        data = np.array(self.buffer.data)

        # select only data columns and convert to float
        return np.array(data[:, self.data_indices], dtype='float64').transpose()

    def cursor_x(self):
        """Current cursor position (x-axis), accounting for downsampling."""
        return self.buffer.cur // self.downsample_factor

    def init_data(self):
        """Initialize the data."""
        channel_data = self.filter(self.current_data())

        for i, _channel in enumerate(self.data_indices):
            data = channel_data[i].tolist()
            self.axes[i].plot(data, linewidth=0.8)
            # plot cursor
            self.axes[i].axvline(self.cursor_x(), color='r')

    def update_buffer(self, fast_forward=False):
        """Update the buffer with latest data and return the data"""
        try:
            records = self.data_source.next_n(
                self.records_per_refresh, fast_forward=fast_forward)
            for row in records:
                self.buffer.append(row)
        except StopIteration:
            self.stop()
        return self.buffer.get()

    def update_view(self, _evt):
        """Called by the timer on refresh."""
        self.update_buffer()
        channel_data = self.filter(self.current_data())

        # plot each channel
        for i, _channel in enumerate(self.data_indices):
            data = channel_data[i].tolist()
            self.axes[i].lines[0].set_ydata(data)
            # cursor line
            self.axes[i].lines[1].set_xdata(self.cursor_x())
            if self.autoscale:
                data_min = min(data)
                data_max = max(data)
                self.axes[i].set_ybound(lower=data_min, upper=data_max)
            else:
                # lower=min(data), upper=max(data))
                self.axes[i].set_ybound(lower=self.y_min, upper=self.y_max)

        self.canvas.draw()


def lsl_data():
    """Constructs an LslDataSource"""
    from bcipy.gui.viewer.lsl_data_source import LslDataSource
    data_source = LslDataSource(stream_type='EEG')
    return (data_source, data_source.device_info, None)


def file_data(path):
    """Constructs a FileStream DataSource"""
    from bcipy.gui.viewer.file_streamer import FileStreamer
    # read metadata
    with open(path) as csvfile:
        r1 = next(csvfile)
        name = r1.strip().split(",")[1]
        r2 = next(csvfile)
        freq = float(r2.strip().split(",")[1])

        reader = csv.reader(csvfile)
        channels = next(reader)
    queue = Queue()
    streamer = FileStreamer(path, queue)
    data_source = QueueDataSource(queue)
    device_info = DeviceInfo(fs=freq, channels=channels, name=name)
    streamer.start()

    return (data_source, device_info, streamer)


def main(data_file: str, seconds: int, downsample_factor: int, refresh: int, yscale: int):
    """Run the viewer gui

    Parameters:
    -----------
        data_file - raw_data.csv file to stream.
        seconds - how many seconds worth of data to display.
        downsample_factor - how much the data is downsampled. A factor of 1
            displays the raw data.
    """
    data_source, device_info, proc = file_data(
        data_file) if data_file else lsl_data()

    app = wx.App(False)
    frame = EegFrame(data_source, device_info,
                     seconds, downsample_factor, refresh, yscale)

    if wx.Display.GetCount() > 1:
        # place frame in the second monitor if one exists.
        s2_x, s2_y, _width, _height = wx.Display(1).GetGeometry()
        offset = 30
        frame.SetPosition((s2_x + offset, s2_y + offset))

    frame.Show(True)
    app.MainLoop()
    if proc:
        proc.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--file', help='path to the data file', default=None)
    parser.add_argument('-s', '--seconds',
                        help='seconds to display', default=5, type=int)
    parser.add_argument('-d', '--downsample',
                        help='downsample factor', default=2, type=int)
    parser.add_argument('-r', '--refresh',
                        help='refresh rate in ms', default=500, type=int)
    parser.add_argument('-y', '--yscale',
                        help='yscale', default=150, type=int)

    args = parser.parse_args()
    main(args.file, args.seconds, args.downsample, args.refresh, args.yscale)
