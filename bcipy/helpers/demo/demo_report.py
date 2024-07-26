from pathlib import Path
from bcipy.helpers.load import load_json_parameters, load_raw_data, load_experimental_data
from bcipy.helpers.triggers import trigger_decoder, TriggerType
from bcipy.config import (
    BCIPY_ROOT,
    DEFAULT_PARAMETER_FILENAME,
    RAW_DATA_FILENAME,
    TRIGGER_FILENAME,
    DEFAULT_DEVICE_SPEC_FILENAME)

from bcipy.acquisition import devices
from bcipy.signal.evaluate.artifact import ArtifactDetection
from bcipy.helpers.report import Report, SignalReportSection, SessionReportSection


 


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


    positions = None
    for session in Path(path).iterdir():
        # loop through the sessions, pausing after each one to allow for manual stopping
        if session.is_dir():
            print(f'Processing {session}')
            prompt = input(f'Hit enter to continue or type "skip" to skip processing: ')
            if prompt != 'skip':
                # load the parameters from the data directory
                parameters = load_json_parameters(
                    f'{session}/{DEFAULT_PARAMETER_FILENAME}', value_cast=True)

                # load the raw data from the data directory
                raw_data = load_raw_data(Path(session, f'{RAW_DATA_FILENAME}.csv'))
                type_amp = raw_data.daq_type

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