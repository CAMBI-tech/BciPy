"""Demonstrates BciPy visualize ERP functionality!

To use at bcipy root,

    - `python bcipy/helpers/demo/demo_visualization.py"`: 
        this will prompt for a BciPy data folder with raw_data.csv, parameters.json and triggers.txt

    - `python bcipy/helpers/demo/demo_visualization.py -p "path://to/bcipy/data/folder"`

    - `python bcipy/helpers/demo/demo_visualization.py --show"`
        this will show the visualization generated in closeable windows

    - - `python bcipy/helpers/demo/demo_visualization.py --save"`
        this will save the visualizations generated to the provided or selected path
"""
from bcipy.helpers.acquisition import analysis_channels
from bcipy.signal.process import get_default_transform
from bcipy.helpers.load import (
    load_experimental_data,
    load_json_parameters,
    load_raw_data,
)
from bcipy.helpers.visualization import visualize_erp
from bcipy.helpers.triggers import TriggerType, trigger_decoder


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--path',
        help='Path to the directory with raw_data to be converted',
        required=False)
    parser.add_argument("--show", dest="show", action="store_true")
    parser.add_argument("--save", dest="save", action="store_true")
    parser.set_defaults(show=False)
    parser.set_defaults(save=False)
    args = parser.parse_args()

    path = args.path
    if not path:
        path = load_experimental_data()

    parameters = load_json_parameters(f'{path}/parameters.json', value_cast=True)

    # extract all relevant parameters
    poststim_length = parameters.get("trial_length")
    prestim_length = parameters.get("prestim_length")
    trials_per_inquiry = parameters.get("stim_length")
    # The task buffer length defines the min time between two inquiries
    # We use half of that time here to buffer during transforms
    buffer = int(parameters.get("task_buffer_length") / 2)
    triggers_file = parameters.get("trigger_file_name", "triggers")
    raw_data_file = parameters.get("raw_data_name", "raw_data.csv")
    # get signal filtering information
    downsample_rate = parameters.get("down_sampling_rate")
    notch_filter = parameters.get("notch_filter_frequency")
    filter_high = parameters.get("filter_high")
    filter_low = parameters.get("filter_low")
    filter_order = parameters.get("filter_order")
    static_offset = parameters.get("static_trigger_offset")
    raw_data = load_raw_data(Path(path, raw_data_file))
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    sample_rate = raw_data.sample_rate

    # setup filtering
    default_transform = get_default_transform(
        sample_rate_hz=sample_rate,
        notch_freq_hz=notch_filter,
        bandpass_low=filter_low,
        bandpass_high=filter_high,
        bandpass_order=filter_order,
        downsample_factor=downsample_rate,
    )
    # Process triggers.txt files
    trigger_targetness, trigger_timing, trigger_symbols = trigger_decoder(
        offset=static_offset,
        trigger_path=f"{path}/{triggers_file}.txt",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
    )
    labels = [0 if label == 'nontarget' else 1 for label in trigger_targetness]
    channel_map = analysis_channels(channels, type_amp)

    figure_handles = visualize_erp(
        raw_data,
        channel_map,
        trigger_timing,
        labels,
        poststim_length,
        transform=default_transform,
        plot_average=True,
        plot_topomaps=True,
        save_path=args.save,
        show=args.show
    )
