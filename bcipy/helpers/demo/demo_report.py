from pathlib import Path
from bcipy.io.load import load_json_parameters, load_raw_data, load_experimental_data
from bcipy.data.triggers import trigger_decoder, TriggerType
from bcipy.config import (
    BCIPY_ROOT,
    DEFAULT_PARAMETERS_FILENAME,
    RAW_DATA_FILENAME,
    TRIGGER_FILENAME,
    DEFAULT_DEVICE_SPEC_FILENAME)

from bcipy.acquisition import devices
from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.visualization import visualize_erp
from bcipy.signal.process import get_default_transform
from bcipy.signal.evaluate.artifact import ArtifactDetection
from bcipy.data.report import Report, SignalReportSection, SessionReportSection


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--path',
        help='Path to the directory with >= 1 sessions to be analyzed for artifacts',
        required=False)

    args = parser.parse_args()
    colabel = True
    # if no path is provided, prompt for one using a GUI
    path = args.path
    if not path:
        path = load_experimental_data()

    trial_window = (0, 1.0)

    positions = None
    for session in Path(path).iterdir():
        # loop through the sessions, pausing after each one to allow for manual stopping
        if session.is_dir():
            print(f'Processing {session}')
            prompt = input('Hit enter to continue or type "skip" to skip processing: ')
            if prompt != 'skip':
                # load the parameters from the data directory
                parameters = load_json_parameters(
                    f'{session}/{DEFAULT_PARAMETERS_FILENAME}', value_cast=True)

                # load the raw data from the data directory
                raw_data = load_raw_data(Path(session, f'{RAW_DATA_FILENAME}.csv'))
                type_amp = raw_data.daq_type
                channels = raw_data.channels
                sample_rate = raw_data.sample_rate
                downsample_rate = parameters.get("down_sampling_rate")
                notch_filter = parameters.get("notch_filter_frequency")
                filter_high = parameters.get("filter_high")
                filter_low = parameters.get("filter_low")
                filter_order = parameters.get("filter_order")
                static_offset = parameters.get("static_trigger_offset")

                default_transform = get_default_transform(
                    sample_rate_hz=sample_rate,
                    notch_freq_hz=notch_filter,
                    bandpass_low=filter_low,
                    bandpass_high=filter_high,
                    bandpass_order=filter_order,
                    downsample_factor=downsample_rate,
                )

                # load the triggers
                if colabel:
                    trigger_type, trigger_timing, trigger_label = trigger_decoder(
                        offset=parameters.get('static_trigger_offset'),
                        trigger_path=f"{session}/{TRIGGER_FILENAME}",
                        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
                    )
                    triggers = (trigger_type, trigger_timing, trigger_label)
                else:
                    triggers = None

                devices.load(Path(BCIPY_ROOT, DEFAULT_DEVICE_SPEC_FILENAME))
                device_spec = devices.preconfigured_device(raw_data.daq_type)
                channel_map = analysis_channels(channels, device_spec)

                # check the device spec for any frontal channels to use for EOG detection
                eye_channels = []
                for channel in device_spec.channels:
                    if 'F' in channel:
                        eye_channels.append(channel)
                if len(eye_channels) == 0:
                    eye_channels = None

                artifact_detector = ArtifactDetection(
                    raw_data,
                    parameters,
                    device_spec,
                    eye_channels=eye_channels,
                    session_triggers=triggers)

                detected = artifact_detector.detect_artifacts()
                figure_handles = visualize_erp(
                    raw_data,
                    channel_map,
                    trigger_timing,
                    trigger_label,
                    trial_window,
                    transform=default_transform,
                    plot_average=True,
                    plot_topomaps=True,
                )

                # Try to find a pkl file in the session folder
                pkl_file = None
                for file in session.iterdir():
                    if file.suffix == '.pkl':
                        pkl_file = file
                        break

                if pkl_file:
                    auc = pkl_file.stem.split('_')[-1]
                else:
                    auc = 'No Signal Model found in session folder'

                sr = SignalReportSection(figure_handles, artifact_detector)
                report = Report(session)
                session = {'Label': 'Demo Session Report', 'AUC': auc}
                session_text = SessionReportSection(session)
                report.add(session_text)
                report.add(sr)
                report.compile()
                report.save()
