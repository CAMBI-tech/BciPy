# Signal Viewer

Th data_viewer GUI allows users to monitor EEG signals during an experiment to ensure that device connections are stable for consistent data quality.

## Code Organization

The main entry point for the code is data\_viewer.py. The data\_viewer is pluggable, and can accept a variety of different DataSources including a data source for streaming from a raw\_data.csv file and an LSL data source for viewing live data streamed during an experiment. The datasource-related code is located in the data\_source subdirectory.

## Usage

### Task monitoring

The primary usage of the viewer is to monitor signal quality during BCI task execution. There is a configuration parameter (acq\_show\_viewer) to control whether or not the viewer is initialized during task startup. This parameter can be set either through the UI or by editing the paramters.json file. If selected, the Viewer will launch during the initialization of the data acquisition module in a new GUI window. The module detects the usage of multiple monitors and will appear in the secondary monitor so that it does not interfere with the main experiment.

By default all active channels will be displayed. However, the Viewer has controls to toggle the visibility of any channel and limit the display to a given montage. Channel information is provided to the viewer through metadata, so it can work for any device supported by BciPy.

There are additional controls for the duration of data to display and to toggle any filtering. Viewing can be paused at any time. Restarting from a paused state will refresh the display with the most recent data.

### Data replay

In addition to its use during an experiment, the Signal Viewer can be run from the command line to replay a raw data file captured during a BciPy session. This modality exposes some additional options for usage.

    (venv) BciPy$ python bcipy/gui/viewer/data\_viewer.py -h
    usage: data\_viewer.py [-h] [-f FILE] [-s SECONDS] [-d DOWNSAMPLE] [-r REFRESH]
    
    optional arguments:
      -h, --help            show this help message and exit
      -f FILE, --file FILE  path to the data file
      -s SECONDS, --seconds SECONDS
                            seconds to display
      -r REFRESH, --refresh REFRESH
                            refresh rate in ms                       

## Architecture

The Signal Viewer has a modular architecture, which gives it a great deal of flexibility. The GUI is implemented as a WxPython Frame that streams data from any object that implements a DataSource interface. The Viewer is also parameterized with a DeviceInfo object which provides information regarding the channels to use and the sample rate. Internally, it uses this metadata to determine how frequently to query the data sources for new data and what channel information to expect.

Several data sources are provided in the module, including a LabStreamingLayer (LSL) data source and a FileDatasource. The LSL datasource is used during the experiment while the FileDataSource is used for data replay purposes. Additionally, the viewer module integrates with the data acquisition module by implementing a custom Processor that starts up the viewer in its own Process.
