"""EEG Data Viewer"""
import sys
from functools import partial
from queue import Queue
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.ticker as ticker
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QTimer  # pylint: disable=no-name-in-module
# pylint: disable=no-name-in-module
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QHBoxLayout,
                             QLabel, QPushButton, QSpinBox, QVBoxLayout,
                             QWidget)

from bcipy.acquisition.devices import DeviceSpec
from bcipy.acquisition.util import StoppableProcess
from bcipy.gui.main import static_text_control
from bcipy.gui.viewer.data_source.data_source import QueueDataSource
from bcipy.gui.viewer.data_source.file_streamer import FileStreamer
from bcipy.gui.viewer.data_source.lsl_data_source import LslDataSource
from bcipy.gui.viewer.ring_buffer import RingBuffer
from bcipy.helpers.parameters import DEFAULT_PARAMETERS_PATH, Parameters
from bcipy.helpers.raw_data import settings
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
        if all_channels[i] not in removed_channels and
        'TRG' not in all_channels[i]
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


class FixedHeightHBox(QWidget):
    """Container for holding controls for the EEG Viewer. Acts like a
    QHBoxLayout with a fixed height."""

    def __init__(self, height: int = 30):
        super().__init__()
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        self.setFixedHeight(height)

    def addWidget(self, widget: QWidget):
        """Add the given widget to the layout"""
        self.layout.addWidget(widget)


class ChannelControls(QWidget):
    """Controls for toggling channels"""

    def __init__(self, font_size: int, channels: List[str],
                 active_channel_indices: List[int],
                 toggle_channel_fn: Callable):
        super().__init__()
        control_stylesheet = f"font-size: {font_size}px;"
        channel_box = QHBoxLayout()
        channel_box.setContentsMargins(0, 0, 0, 0)

        for channel_index in active_channel_indices:
            channel_name = channels[channel_index]
            chkbox = QCheckBox(channel_name)
            chkbox.setChecked(True)
            chkbox.setStyleSheet(control_stylesheet)
            chkbox.toggled.connect(partial(toggle_channel_fn, channel_index))
            channel_box.addWidget(chkbox)
        self.setLayout(channel_box)
        self.setFixedHeight(font_size + 4)


class FixedScaleInput(QWidget):
    """Input for adjusting the fixed scale value"""

    def __init__(self,
                 initial_value: int,
                 on_change_fn: Callable,
                 label: str = 'Fixed scale:',
                 max_value: int = 5000,
                 font_size: int = 11):
        super().__init__()
        stylesheet = f"font-size: {font_size}px;"
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel(label)
        lbl.setStyleSheet(stylesheet)
        lbl.setAlignment(Qt.AlignRight)
        lbl.setContentsMargins(0, 2, 0, 0)
        layout.addWidget(lbl)

        self.fixed_scale_input = QSpinBox()
        self.fixed_scale_input.setMaximum(max_value)
        self.fixed_scale_input.setValue(initial_value)
        self.fixed_scale_input.setStyleSheet(stylesheet)
        self.fixed_scale_input.setAlignment(Qt.AlignLeft)
        self.fixed_scale_input.valueChanged.connect(on_change_fn)
        self.fixed_scale_input.setFocusPolicy(Qt.ClickFocus)

        layout.addWidget(self.fixed_scale_input)
        self.setLayout(layout)

    def value(self) -> int:
        """Returns the input value"""
        return self.fixed_scale_input.value()


class EEGPanel(QWidget):
    """GUI Frame in which data is plotted. Plots a subplot for every channel.
    Relies on a Timer to retrieve data at a specified interval. Data to be
    displayed is retrieved from a provided DataSource.

    Parameters:
    -----------
    - data_source : object that implements the viewer DataSource interface.
    - device_spec : metadata about the data.
    - parameters : configuration for filters, etc.
    - seconds : how many seconds worth of data to display.
    - refresh : time in milliseconds; how often to refresh the plots
    - y_scale : max y-value to use when using a fixed scale for plots
    (autoscale turned off);
    """
    control_font_size = 11
    # space between axis label and tick labels
    yaxis_label_space = 60
    yaxis_label_fontsize = 12
    # fixed width font so we can adjust spacing predictably
    yaxis_tick_font = 'DejaVu Sans Mono'
    yaxis_tick_fontsize = 10

    def __init__(self,
                 data_source,
                 device_spec: DeviceSpec,
                 parameters: Parameters,
                 seconds: int = 5,
                 refresh: int = 500,
                 y_scale=500):
        super().__init__()

        self.data_source = data_source
        self.parameters = parameters
        self.refresh_rate = refresh
        self.samples_per_second = device_spec.sample_rate
        self.records_per_refresh = int(
            (self.refresh_rate / 1000) * self.samples_per_second)

        self.channels = device_spec.channels
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

        # The buffer stores raw, unfiltered data.
        self.buffer = init_buffer(self.samples_per_second, self.seconds,
                                  self.channels)
        self.init_data_plots()
        self.axes_changed = False

    # pylint: disable=attribute-defined-outside-init
    def init_canvas(self):
        """Initialize the Figure for drawing plots"""
        self.y_min = -self.y_scale
        self.y_max = self.y_scale

        # figure size is in inches.
        self.figure = Figure(figsize=(12, 9),
                             dpi=72,
                             tight_layout={'pad': 0.0})

        self.axes = self.init_axes()
        self.axes_bounds = self.init_axes_bounds()
        self.canvas = FigureCanvasQTAgg(self.figure)

    # pylint: disable=invalid-name,attribute-defined-outside-init
    def initUI(self):
        """Initialize the UI"""
        vbox = QVBoxLayout()

        self.init_canvas()
        vbox.addWidget(self.canvas)

        # Toolbar
        self.toolbar = QVBoxLayout()

        controls = FixedHeightHBox()

        control_stylesheet = f"font-size: {self.control_font_size}px;"

        # Start/Pause button
        self.start_stop_btn = QPushButton('Pause' if self.started else 'Start',
                                          self)
        self.start_stop_btn.setFixedWidth(80)
        self.start_stop_btn.setStyleSheet(control_stylesheet)
        self.start_stop_btn.clicked.connect(self.toggle_stream)
        controls.addWidget(self.start_stop_btn)

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
        self.seconds_input.setStyleSheet(control_stylesheet)
        self.seconds_input.currentIndexChanged.connect(self.seconds_handler)
        controls.addWidget(self.seconds_input)

        # Autoscale checkbox
        self.autoscale_checkbox = QCheckBox('Auto-scale')
        self.autoscale_checkbox.setStyleSheet(control_stylesheet)
        self.autoscale_checkbox.setChecked(self.autoscale)
        self.autoscale_checkbox.toggled.connect(self.toggle_autoscale_handler)
        controls.addWidget(self.autoscale_checkbox)

        # Fixed scale input
        self.fixed_scale_input = FixedScaleInput(
            self.y_scale,
            self.fixed_scale_handler,
            label='Fixed scale:',
            font_size=self.control_font_size)
        controls.addWidget(self.fixed_scale_input)

        # Filter checkbox
        self.sigpro_checkbox = QCheckBox('Filtered')
        self.sigpro_checkbox.setStyleSheet(control_stylesheet)
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
            size=10,
            color='dimgray')
        self.filter_settings_text.setWordWrap(False)
        controls.addWidget(self.filter_settings_text)

        self.toolbar.addWidget(controls)

        channel_box = ChannelControls(self.control_font_size, self.channels,
                                      self.active_channel_indices,
                                      self.toggle_channel)
        self.toolbar.addWidget(channel_box)

        vbox.addLayout(self.toolbar)

        self.setWindowTitle('EEG Viewer')
        self.setLayout(vbox)
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
        self.axes_bounds = self.init_axes_bounds()
        self.init_data_plots()

    def init_axes_bounds(self):
        """Initial value for ymin, ymax bounds for every axis."""
        return [None] * len(self.axes)

    def toggle_channel(self, channel_index):
        """Remove the provided channel from the display"""
        channel = self.channels[channel_index]
        if self.started:
            self.stop()

        if channel in self.removed_channels:
            self.removed_channels.remove(channel)
        else:
            self.removed_channels.append(channel)
        self.active_channel_indices = active_indices(self.channels,
                                                     self.removed_channels)
        self.axes_changed = True

    def start(self):
        """Start streaming data in the viewer."""
        # update buffer with latest data on (re)start.
        self.start_stop_btn.setText('Pause')
        stylesheet = f'font-size: {self.control_font_size}px;'
        self.start_stop_btn.setStyleSheet(stylesheet)
        self.start_stop_btn.repaint()
        self.started = True

        if self.axes_changed:
            self.axes_changed = False
            self.reset_axes()

        self.update_buffer(fast_forward=True)
        self.timer.start(self.refresh_rate)

    def stop(self):
        """Stop/Pause the viewer."""
        self.start_stop_btn.setText('Start')
        stylesheet = (f'font-size: {self.control_font_size}px;'
                      'color: white; background-color: green;'
                      'border: 1px solid darkgreen;'
                      'border-radius: 4px; padding: 3px;')
        self.start_stop_btn.setStyleSheet(stylesheet)
        self.start_stop_btn.repaint()
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

    def fixed_scale_handler(self):
        """Event handler for updating the fixed scale"""
        self.with_refresh(self.update_fixed_scale)

    def update_fixed_scale(self):
        """Sets the fixed scale value from the input."""
        self.y_scale = abs(self.fixed_scale_input.value())
        self.y_min = -self.y_scale
        self.y_max = self.y_scale
        self.autoscale = False
        self.autoscale_checkbox.setChecked(self.autoscale)
        self.autoscale_checkbox.repaint()

    def with_refresh(self, fn):
        """Pauses streaming, performs the given action, and sets a flag
        indicating that the display axes should be refreshed on restart."""

        if self.started:
            self.stop()
        fn()
        self.axes_changed = True

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
            # Initial data plotted will be a list 0.0; this will be updated
            # with real data every tick using the `update_plots` method.
            data = channel_data[i].tolist()
            self.axes[i].set_ybound(lower=self.y_min, upper=self.y_max)
            self.axes[i].plot(data, linewidth=0.8)
            # plot cursor
            self.axes[i].axvline(cursor_x, color='r')

    def update_buffer(self, fast_forward: bool = False) -> None:
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

    def update_axes_bounds(self, index: int, data: List[float]) -> bool:
        """Update the upper and lower bounds of the axis at the given index.

        Parameters
        ----------
        - index : index of the axis to update
        - data : current data to be displayed

        Returns
        -------
        bool indicating whether or not the bounds changed
        """
        assert self.axes_bounds, "Axes bounds must be initialized"
        # Maintains a symmetrical scale. Alternatively we could set:
        #   data_min = round(min(data))
        #   data_max = round(max(data))
        data_max = max(abs(round(min(data))), abs(round(max(data))))
        data_min = -data_max
        if self.axes_bounds[index]:
            current_min, current_max = self.axes_bounds[index]
            self.axes_bounds[index] = (min(current_min, data_min),
                                       max(current_max, data_max))
            return (current_min, current_max) != self.axes_bounds[index]
        self.axes_bounds[index] = (data_min, data_max)
        return True

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
                bounds_changed = self.update_axes_bounds(i, data)
                data_min, data_max = self.axes_bounds[i]

                self.axes[i].set_ybound(lower=data_min, upper=data_max)

                if bounds_changed:
                    # For ylabels to be aligned consistently, labelpad is re-calculated
                    ch_name = self.channels[_channel]
                    tick_labels = self.axes[i].get_yticks()
                    # Min tick value does not display so index is 1, not 0.
                    pad = self.adjust_padding(data_min, data_max)
                    self.axes[i].set_ylabel(ch_name,
                                            rotation=0,
                                            labelpad=pad,
                                            fontsize=12)
            else:
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


def lsl_data() -> Tuple[LslDataSource, DeviceSpec, None]:
    """Constructs an LslDataSource, which provides data written to an LSL EEG
    stream."""
    data_source = LslDataSource(stream_type='EEG')
    return (data_source, data_source.device_spec, None)


def file_data(path: str
              ) -> Tuple[QueueDataSource, DeviceSpec, StoppableProcess]:
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
    # read metadata
    name, freq, channels = settings(path)
    queue = Queue()
    streamer = FileStreamer(path, queue)
    data_source = QueueDataSource(queue)
    device_spec = DeviceSpec(name=name, channels=channels, sample_rate=freq)
    streamer.start()

    return (data_source, device_spec, streamer)


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
    data_source, device_spec, proc = file_data(
        data_file) if data_file else lsl_data()

    app = QApplication(sys.argv)
    panel = EEGPanel(data_source, device_spec,
                     Parameters(parameters, cast_values=True), seconds,
                     refresh, yscale)

    if display_screen == 1 and len(app.screens()) > 1:
        # place frame in the second monitor if one exists
        non_primary_screens = [
            screen for screen in app.screens()
            if screen != app.primaryScreen()
        ]
        display_monitor = non_primary_screens[0]
        monitor = display_monitor.geometry()
    else:
        monitor = app.primaryScreen().geometry()

    # increase height to 90% of monitor height and preserve aspect ratio.
    new_height = int(monitor.height() * 0.9)
    pct_increase = (new_height - panel.height()) / panel.height()
    new_width = panel.width() + int(panel.width() * pct_increase)

    panel.resize(new_width, new_height)
    panel.move(monitor.left(), monitor.top())

    panel.start()

    app_exit = app.exec_()
    if proc:
        proc.stop()
    sys.exit(app_exit)


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
    parser.add_argument('-y', '--yscale', help='yscale', default=500, type=int)
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
