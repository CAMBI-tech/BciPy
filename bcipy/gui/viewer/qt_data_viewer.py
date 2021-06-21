"""EEG Data Viewer"""
import csv
import sys
from functools import partial
from queue import Queue
from typing import Callable, Dict, List, Optional, Tuple

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import numpy as np
from PyQt5.QtCore import QTimer # pylint: disable=no-name-in-module
# pylint: disable=no-name-in-module
from PyQt5.QtWidgets import (QApplication, QDesktopWidget, QCheckBox, QComboBox, QHBoxLayout,
                             QPushButton, QVBoxLayout, QWidget)

from bcipy.acquisition.device_info import DeviceInfo
from bcipy.gui.gui_main import static_text_control
from bcipy.gui.viewer.data_source.data_source import QueueDataSource
from bcipy.gui.viewer.ring_buffer import RingBuffer
from bcipy.helpers.parameters import DEFAULT_PARAMETERS_PATH, Parameters
from bcipy.signal.process.transform import Downsample, get_default_transform


def filters(
        sample_rate_hz: float, parameters: Parameters
) -> Dict[str, Callable[[np.ndarray, Optional[int]], Tuple[np.ndarray, int]]]:
    """Returns a dict of filters that can be used"""
    return {
        'downsample':
        Downsample(parameters['down_sampling_rate']),
        'default_transform':
        get_default_transform(
            sample_rate_hz=sample_rate_hz,
            notch_freq_hz=parameters['notch_filter_frequency'],
            bandpass_low=parameters['filter_low'],
            bandpass_high=parameters['filter_high'],
            bandpass_order=parameters['filter_order'],
            downsample_factor=parameters['down_sampling_rate'])
    }


def active_indices(all_channels, removed_channels) -> List[int]:
    """List of indices of all channels which will be displayed. By default
    filters out TRG channels and any channels marked as
    removed_channels.

    Parameters
    ----------
    - all_channels : list of channel names
    - removed_channels : list of channel names to exclude

    Returns
    -------
    list of indices for channel names in use.
    """

    return [
        i for i in range(len(all_channels))
        if all_channels[i] not in removed_channels
        and 'TRG' not in all_channels[i]
    ]


def init_buffer(samples_per_second: int, seconds: int,
                channels: List[str]) -> RingBuffer:
    """Initialize the underlying RingBuffer by pre-allocating empty
    values. Buffer size is determined by the sample frequency and the
    number of seconds to display.

    Parameters
    ----------
    - samples_per_second : sample rate
    - seconds : number of seconds of data to store
    - channels : list of channel names
    """

    buf_size = int(samples_per_second * seconds)
    empty_val = [0.0 for _x in channels]
    return RingBuffer(buf_size, pre_allocated=True, empty_value=empty_val)


class EEGPanel(QWidget):
    """GUI Frame in which data is plotted. Plots a subplot for every channel.
    Relies on a Timer to retrieve data at a specified interval. Data to be
    displayed is retrieved from a provided DataSource.

    Parameters:
    -----------
    - data_source : object that implements the viewer DataSource interface.
    - device_info : metadata about the data.
    - parameters : configuration for filters, etc.
    - seconds : how many seconds worth of data to display.
    - refresh : time in milliseconds; how often to refresh the plots
    """

    def __init__(self,
                 data_source,
                 device_info: DeviceInfo,
                 parameters: Parameters,
                 seconds: int = 5,
                 refresh: int = 500,
                 y_scale=100):
        super().__init__()

        self.data_source = data_source
        self.parameters = parameters
        self.refresh_rate = refresh
        self.samples_per_second = device_info.fs
        self.records_per_refresh = int(
            (self.refresh_rate / 1000) * self.samples_per_second)

        self.channels = device_info.channels
        self.removed_channels = ['TRG', 'timestamp']
        self.active_channel_indices = active_indices(self.channels,
                                                     self.removed_channels)

        self.seconds = seconds
        self.downsample_factor = parameters['down_sampling_rate']

        self.filter_options = filters(self.samples_per_second, parameters)
        self.selected_filter = 'downsample'

        self.autoscale = True
        self.y_scale = y_scale

        self.started = False
        self.initUI()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)

        # Start streaming
        # # The buffer stores raw, unfiltered data.
        self.buffer = init_buffer(self.samples_per_second, self.seconds,
                                  self.channels)
        self.init_data_plots()
        self.start()

    # pylint: disable=attribute-defined-outside-init
    def init_canvas(self):
        """Initialize the Figure for drawing plots"""
        self.y_min = -self.y_scale
        self.y_max = self.y_scale

        # figure size is in inches.
        self.figure = Figure(figsize=(12, 9),
                             dpi=80,
                             tight_layout={'pad': 0.0})
        # space between axis label and tick labels
        self.yaxis_label_space = 60
        self.yaxis_label_fontsize = 14
        # fixed width font so we can adjust spacing predictably
        self.yaxis_tick_font = 'DejaVu Sans Mono'
        self.yaxis_tick_fontsize = 10

        self.axes = self.init_axes()
        self.canvas = FigureCanvasQTAgg(self.figure)

    # pylint: disable=invalid-name,attribute-defined-outside-init
    def initUI(self):
        """Initialize the UI"""
        vbox = QVBoxLayout()

        self.init_canvas()
        vbox.addWidget(self.canvas)

        # Toolbar
        self.toolbar = QVBoxLayout()

        controls = QHBoxLayout()

        # Start/Pause button
        self.start_stop_btn = QPushButton('Pause', self)
        self.start_stop_btn.setFixedWidth(80)
        self.start_stop_btn.clicked.connect(self.toggle_stream)
        controls.addWidget(self.start_stop_btn)

        # Autoscale checkbox
        self.autoscale_checkbox = QCheckBox('Auto-scale')
        self.autoscale_checkbox.setChecked(self.autoscale)
        self.autoscale_checkbox.toggled.connect(self.toggle_autoscale_handler)
        controls.addWidget(self.autoscale_checkbox)

        # Pulldown list of seconds to display
        self.seconds_choices = [2, 5, 10]
        if self.seconds not in self.seconds_choices:
            self.seconds_choices.append(self.seconds)
            self.seconds_choices.sort()

        self.seconds_input = QComboBox()
        self.seconds_input.addItems(
            [str(x) + " seconds" for x in self.seconds_choices])
        self.seconds_input.setCurrentIndex(
            self.seconds_choices.index(self.seconds))
        self.seconds_input.currentIndexChanged.connect(self.seconds_handler)
        controls.addWidget(self.seconds_input)

        # Filter checkbox
        self.sigpro_checkbox = QCheckBox('Filtered')
        self.sigpro_checkbox.setChecked(False)
        self.sigpro_checkbox.toggled.connect(self.toggle_filtering_handler)
        controls.addWidget(self.sigpro_checkbox)

        # Filter settings label
        filter_settings = [
            f"Downsample: {self.downsample_factor}",
            f"Notch Freq: {self.parameters['notch_filter_frequency']}",
            f"Low: {self.parameters['filter_low']}",
            f"High: {self.parameters['filter_high']}",
            f"Order: {self.parameters['filter_order']}"
        ]
        self.filter_settings_text = static_text_control(
            self,
            label=f"[{', '.join(filter_settings)}]",
            size=11,
            color='dimgray')
        controls.addWidget(self.filter_settings_text)

        # Buttons for toggling channels
        channel_box = QHBoxLayout()
        for channel_index in self.active_channel_indices:
            channel_name = self.channels[channel_index]
            chkbox = QCheckBox(channel_name)
            chkbox.setChecked(channel_name not in self.removed_channels)
            chkbox.toggled.connect(partial(self.toggle_channel, channel_index))
            channel_box.addWidget(chkbox)

        self.toolbar.addLayout(controls)
        self.toolbar.addLayout(channel_box)

        vbox.addLayout(self.toolbar)

        self.setWindowTitle('EEG Viewer')
        self.setLayout(vbox)
        self.setMinimumWidth(800)
        self.setMinimumHeight(550)
        self.show()

    @property
    def filter(self):
        """Returns the current filter"""
        return self.filter_options[self.selected_filter]

    def init_axes(self):
        """Sets configuration for axes. Creates a subplot for every data
        channel and configures the appropriate labels and tick marks."""

        axes = self.figure.subplots(len(self.active_channel_indices),
                                    1,
                                    sharex=True)
        for i, channel in enumerate(self.active_channel_indices):
            ch_name = self.channels[channel]
            axes[i].set_frame_on(False)
            axes[i].set_ylabel(ch_name,
                               rotation=0,
                               labelpad=self.yaxis_label_space,
                               fontsize=self.yaxis_label_fontsize)
            # x-axis shows seconds in 0.5 sec increments
            tick_names = np.arange(0, self.seconds, 0.5)
            ticks = [(self.samples_per_second * sec) / self.downsample_factor
                     for sec in tick_names]
            axes[i].xaxis.set_major_locator(ticker.FixedLocator(ticks))
            axes[i].xaxis.set_major_formatter(
                ticker.FixedFormatter(tick_names))
            axes[i].tick_params(axis='y',
                                which='major',
                                labelsize=self.yaxis_tick_fontsize)
            for tick in axes[i].get_yticklabels():
                tick.set_fontname(self.yaxis_tick_font)
            axes[i].grid()
        return axes

    def reset_axes(self):
        """Clear the data in the GUI."""
        self.figure.clear()
        self.axes = self.init_axes()

    def toggle_channel(self, channel_index):
        """Remove the provided channel from the display"""
        channel = self.channels[channel_index]
        previously_running = self.started
        if self.started:
            self.stop()

        if channel in self.removed_channels:
            self.removed_channels.remove(channel)
        else:
            self.removed_channels.append(channel)
        self.active_channel_indices = active_indices(self.channels,
                                                     self.removed_channels)
        self.reset_axes()
        self.init_data_plots()
        self.canvas.draw()
        if previously_running:
            self.start()

    def start(self):
        """Start streaming data in the viewer."""
        # update buffer with latest data on (re)start.
        self.start_stop_btn.setText('Pause')
        self.started = True
        self.update_buffer(fast_forward=True)
        self.timer.start(self.refresh_rate)

    def stop(self):
        """Stop/Pause the viewer."""
        self.start_stop_btn.setText('Start')
        self.started = False
        self.timer.stop()

    def toggle_stream(self):
        """Toggle data streaming"""
        if self.started:
            self.stop()
        else:
            self.start()

    def toggle_filtering_handler(self):
        """Event handler for toggling data filtering."""
        self.with_refresh(self.toggle_filtering)

    def toggle_filtering(self):
        """Toggles data filtering."""
        self.selected_filter = 'default_transform' if self.sigpro_checkbox.isChecked(
        ) else 'downsample'

    def toggle_autoscale_handler(self):
        """Event handler for toggling autoscale"""
        self.with_refresh(self.toggle_autoscale)

    def toggle_autoscale(self):
        """Sets autoscale to checkbox value"""
        self.autoscale = self.autoscale_checkbox.isChecked()

    def seconds_handler(self):
        """Event handler for changing seconds"""
        self.with_refresh(self.update_seconds)

    def update_seconds(self):
        """Set the number of seconds worth of data to display from the
        pulldown list."""
        self.seconds = self.seconds_choices[self.seconds_input.currentIndex()]
        self.buffer = init_buffer(self.samples_per_second, self.seconds,
                                  self.channels)

    def with_refresh(self, fn):
        """Performs the given action and refreshes the display."""
        previously_running = self.started
        if self.started:
            self.stop()
        fn()
        # re-initialize
        self.reset_axes()
        self.init_data_plots()
        if previously_running:
            self.start()

    @property
    def current_data(self):
        """Returns the data as an np array with a row of floats for each
        displayed channel."""

        # array of 'object'; TRG data may be strings
        data = np.array(self.buffer.data)

        # select only data columns and convert to float
        return np.array(data[:, self.active_channel_indices],
                        dtype='float64').transpose()

    def filtered_data(self):
        """Channel data after applying the current filter.

        Returns the data as an np array with a row of floats for each
        displayed channel.
        """
        channel_data, _fs = self.filter(self.current_data,
                                        self.samples_per_second)
        return channel_data

    @property
    def cursor_x(self):
        """Current cursor position (x-axis), accounting for downsampling."""
        return self.buffer.cur // self.downsample_factor

    def init_data_plots(self):
        """Initialize the data in the viewer."""
        channel_data = self.filtered_data()

        cursor_x = self.cursor_x
        for i, _channel in enumerate(self.active_channel_indices):
            data = channel_data[i].tolist()
            self.axes[i].plot(data, linewidth=0.8)
            # plot cursor
            self.axes[i].axvline(cursor_x, color='r')

    def update_buffer(self, fast_forward=False):
        """Update the buffer with latest data from the datasource.
        If the datasource does not have the requested number of
        samples, viewer streaming is stopped."""
        try:
            records = self.data_source.next_n(self.records_per_refresh,
                                              fast_forward=fast_forward)
            for row in records:
                self.buffer.append(row)
        except StopIteration:
            self.stop()
        except BaseException:
            self.stop()

    def update_plots(self):
        """Called by the timer on refresh. Updates the buffer with the latest
        data and refreshes the plots. This is called on every tick."""
        self.update_buffer()
        channel_data = self.filtered_data()

        cursor_x = self.cursor_x
        # plot each channel
        for i, _channel in enumerate(self.active_channel_indices):
            data = channel_data[i].tolist()
            self.axes[i].lines[0].set_ydata(data)
            # cursor line
            self.axes[i].lines[1].set_xdata(cursor_x)
            if self.autoscale:
                data_min = min(data)
                data_max = max(data)
                self.axes[i].set_ybound(lower=data_min, upper=data_max)

                # For ylabels to be aligned consistently, labelpad is
                # re-calculated on every draw.
                ch_name = self.channels[_channel]
                tick_labels = self.axes[i].get_yticks()
                # Min tick value does not display so index is 1, not 0.
                pad = self.adjust_padding(int(tick_labels[1]),
                                          int(tick_labels[-1]))
                self.axes[i].set_ylabel(ch_name,
                                        rotation=0,
                                        labelpad=pad,
                                        fontsize=14)
            else:
                # lower=min(data), upper=max(data))
                self.axes[i].set_ybound(lower=self.y_min, upper=self.y_max)

        self.canvas.draw()

    def adjust_padding(self, data_min: int, data_max: int) -> int:
        """Attempts to keep the channel labels in the same position by adjusting
        the padding between the yaxis label and the yticks."""
        digits_min = len(str(data_min))
        digits_max = len(str(data_max))
        chars = max(digits_min, digits_max)

        # assume at least 2 digits to start.
        baseline_chars = 2
        # Value determined by trial and error; this may change if the tick font
        # or font size is adjusted.
        ytick_digit_width = 7
        return self.yaxis_label_space - \
            ((chars - baseline_chars) * ytick_digit_width)


def lsl_data():
    """Constructs an LslDataSource, which provides data written to an LSL EEG
    stream."""
    from bcipy.gui.viewer.data_source.lsl_data_source import LslDataSource
    data_source = LslDataSource(stream_type='EEG')
    return (data_source, data_source.device_info, None)


def file_data(path: str):
    """Constructs a QueueDataSource from the contents of the file at the given
    path. Data is written to the datasource at the rate specified in the
    raw_data.csv metadata, so it acts as a live stream.

    Parameters
    ----------
        path - path to the raw_data.csv file
    Returns
    -------
        (QueueDataSource, DeviceInfo, FileStreamer)
            - QueueDataSource can be provided to the EEGFrame
            - DeviceInfo provides the metadata read from the raw_data.csv
            - data is not written to the QueueDataSource until the
                FileStreamer (StoppableProcess) is started.
    """

    from bcipy.gui.viewer.data_source.file_streamer import FileStreamer
    # read metadata
    with open(path) as csvfile:
        row1 = next(csvfile)
        name = row1.strip().split(",")[1]
        row2 = next(csvfile)
        freq = float(row2.strip().split(",")[1])

        reader = csv.reader(csvfile)
        channels = next(reader)
    queue = Queue()
    streamer = FileStreamer(path, queue)
    data_source = QueueDataSource(queue)
    device_info = DeviceInfo(fs=freq, channels=channels, name=name)
    streamer.start()

    return (data_source, device_info, streamer)


def main(data_file: str,
         seconds: int,
         refresh: int,
         yscale: int,
         display_screen: int = 1,
         parameters: str = DEFAULT_PARAMETERS_PATH):
    """Run the viewer GUI. If a raw_data.csv data_file is provided, data is
    streamed from that, otherwise it will read from an LSLDataStream by
    default.

    Parameters:
    -----------
        data_file - optional, raw_data.csv file to stream.
        seconds - how many seconds worth of data to display.
        downsample_factor - how much the data is downsampled. A factor of 1
            displays the raw data.
        display_screen - monitor in which to display the viewer
            (0 for primary, 1 for secondary)
        parameters - location of parameters.json file with configuration for filters.
    """
    data_source, device_info, proc = file_data(
        data_file) if data_file else lsl_data()

    app = QApplication(sys.argv)
    panel = EEGPanel(data_source, device_info,
                      Parameters(parameters, cast_values=True), seconds,
                      refresh, yscale)

    if display_screen == 1 and len(app.screens()) > 1:
        # place frame in the second monitor if one exists
        non_primary_screens = [screen for screen in app.screens() if screen != app.primaryScreen()]
        display_monitor = non_primary_screens[0]
        monitor = display_monitor.geometry()
        panel.move(monitor.left(), monitor.top())

    sys.exit(app.exec_())

    if proc:
        proc.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f',
                        '--file',
                        help='path to the data file',
                        default=None)
    parser.add_argument('-p',
                        '--parameters',
                        default=DEFAULT_PARAMETERS_PATH,
                        help='Parameter location. Pass as *.json')
    parser.add_argument('-s',
                        '--seconds',
                        help='seconds to display',
                        default=5,
                        type=int)
    parser.add_argument('-r',
                        '--refresh',
                        help='refresh rate in ms',
                        default=500,
                        type=int)
    parser.add_argument('-y', '--yscale', help='yscale', default=150, type=int)
    parser.add_argument('-m',
                        '--monitor',
                        help='display screen (0: primary, 1: secondary)',
                        default=0,
                        type=int)

    args = parser.parse_args()
    main(data_file=args.file,
         seconds=args.seconds,
         refresh=args.refresh,
         yscale=args.yscale,
         display_screen=args.monitor,
         parameters=args.parameters)
