import subprocess
from typing import Any, Optional
import logging
from pathlib import Path
import glob

from bcipy.gui.experiments.ExperimentField import start_experiment_field_collection_gui
from bcipy.task import Task
from bcipy.helpers.triggers import trigger_decoder, TriggerType

from bcipy.acquisition import devices
from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.parameters import Parameters
from bcipy.acquisition.devices import DeviceSpec
from bcipy.helpers.load import load_raw_data
from bcipy.signal.process import get_default_transform
from bcipy.helpers.report import SignalReportSection, SessionReportSection, Report
from bcipy.task.main import TaskData
from bcipy.config import DEFAULT_PARAMETERS_PATH, SESSION_LOG_FILENAME, RAW_DATA_FILENAME, TRIGGER_FILENAME
from bcipy.signal.model.offline_analysis import offline_analysis
from bcipy.helpers.visualization import visualize_erp
from bcipy.signal.evaluate.artifact import ArtifactDetection


logger = logging.getLogger(SESSION_LOG_FILENAME)


class CodeHookAction(Task):
    """
    Action for running generic code hooks.
    """

    name = "Code Hook Action"

    def __init__(
            self,
            parameters: Parameters,
            data_directory: str,
            code_hook: Optional[str] = None,
            subprocess: bool = True,
            **kwargs) -> None:
        super().__init__()
        self.code_hook = code_hook
        self.subprocess = subprocess

    def execute(self) -> TaskData:
        if self.code_hook:
            if self.subprocess:
                subprocess.Popen(self.code_hook, shell=True)

            else:
                subprocess.run(self.code_hook, shell=True)
        return TaskData()


class OfflineAnalysisAction(Task):
    """
    Action for running offline analysis.
    """

    name = "Offline Analysis Action"

    def __init__(
            self,
            parameters: Parameters,
            data_directory: str,
            parameters_path: str = f'{DEFAULT_PARAMETERS_PATH}',
            last_task_dir: Optional[str] = None,
            alert: bool = False,
            **kwargs: Any) -> None:
        super().__init__()
        self.parameters = parameters
        self.parameters_path = parameters_path
        self.alert_finished = alert

        if last_task_dir:
            self.data_directory = last_task_dir
        else:
            self.data_directory = data_directory

    def execute(self) -> TaskData:
        response = offline_analysis(self.data_directory, self.parameters, alert_finished=self.alert_finished)
        return TaskData(
            save_path=self.data_directory,
            task_dict={"parameters": self.parameters_path,
                       "response": response},
        )


class ExperimentFieldCollectionAction(Task):
    """
    Action for collecting experiment field data.
    """

    name = "Experiment Field Collection Action"

    def __init__(
            self,
            parameters: Parameters,
            save_path: str,
            experiment_id: str = 'default',
            **kwargs: Any) -> None:
        super().__init__()
        self.experiment_id = experiment_id
        self.save_folder = save_path
        self.parameters = parameters

    def execute(self) -> TaskData:
        logger.info(
            f"Collecting experiment field data for experiment {self.experiment_id} in save folder {self.save_folder}"
        )
        start_experiment_field_collection_gui(self.experiment_id, self.save_folder)
        return TaskData(
            save_path=self.save_folder,
            task_dict={
                "experiment_id": self.experiment_id,
            },
        )

class BciPyCalibrationReportAction(Task):
    """
    Action for generating a report.
    """

    name = "BciPy Report Action"

    def __init__(
            self,
            parameters: Parameters,
            save_path: str,
            protocol_path: Optional[str] = None,
            last_task_dir: Optional[str] = None,
            trial_window: Optional[tuple] = None,
            **kwargs: Any) -> None:
        super().__init__()
        self.save_folder = save_path
        self.parameters = parameters
        self.protocol_path = protocol_path
        self.last_task_dir = last_task_dir
        self.default_transform = None
        self.trial_window = trial_window or (0, 1.0)
        self.static_offset = self.parameters.get("static_trigger_offset")
        self.report = Report(self.save_folder)
        self.report_sections = []
        self.all_raw_data = []
        self.type_amp = None

    def execute(self) -> TaskData:
        """Excute the report generation action.
        
        This assumes all data were collected using the same protocol, device, and parameters.
        """
        logger.info(f"Generating report in save folder {self.save_folder}")
        # loop through all the files in the last_task_dir

        # If a protocol is given, loop over and look for any calibration directories
        if self.protocol_path:
            # Use glob to find all directories with Calibration in the name
            data_directories = []
            all_raw_data = []
            for data_dir in glob.glob(f"{self.protocol_path}/**/Calibration", recursive=True):
                data_directories.append(dir)
                # For each calibration directory, attempt to load the raw data
                session_report = self.create_session_report(data_dir)
                self.report_sections.append(session_report)
                signal_report_section = self.create_signal_report(data_dir)
                self.report_sections.append(signal_report_section)
                self.report.add(session_report)
                self.report.add(signal_report_section)

        self.report.compile()
        self.report.save()

    def create_signal_report(self, data_dir) -> SignalReportSection:
        raw_data = load_raw_data(Path(data_dir, f'{RAW_DATA_FILENAME}.csv'))
        if not self.type_amp:
            self.type_amp = raw_data.daq_type
        channels = raw_data.channels
        sample_rate = raw_data.sample_rate
        device_spec = devices.preconfigured_device(raw_data.daq_type)
        channel_map = analysis_channels(channels, device_spec)
        self.all_raw_data.append(raw_data)

        # Set the default transform if not already set
        if not self.default_transform:
            self.set_default_transform(sample_rate)
        
        # get figure handles
        figure_handles = self.get_figure_handles(dir, raw_data, channel_map)
        artifact_detector = self.get_artifact_detector(raw_data, device_spec)
        return SignalReportSection(figure_handles, artifact_detector)

    def create_session_report(self, data_dir) -> SessionReportSection:
        # get task name
        task = self.parameters.get("task", "Unknown")
        summary_dict = {
            "task": task,
            "data_dir": data_dir,
            "type_amp": self.type_amp
        }
        signal_model_metrics = self.get_signal_model_metrics(data_dir)
        summary_dict.update(signal_model_metrics)

        return SessionReportSection(summary_dict)
    
    def get_signal_model_metrics(self, data_directory: Path) -> dict:
        """Get the signal model metrics from the session folder.
        
        In the future, the model will save a ModelMetrics with the pkl file. 
        For now, we just look for the pkl file and extract the AUC from the filename.
        """
        pkl_file = None
        for file in data_directory.iterdir():
            if file.suffix == '.pkl':
                pkl_file = file
                break

        if pkl_file:
            auc = pkl_file.stem.split('_')[-1]
        else:
            auc = 'No Signal Model found in session folder'

        return {'AUC': auc}

    def set_default_transform(self, sample_rate: int) -> None:
        downsample_rate = self.parameters.get("down_sampling_rate")
        notch_filter = self.parameters.get("notch_filter_frequency")
        filter_high = self.parameters.get("filter_high")
        filter_low = self.parameters.get("filter_low")
        filter_order = self.parameters.get("filter_order")
        self.default_transform = get_default_transform(
            sample_rate_hz=sample_rate,
            notch_freq_hz=notch_filter,
            bandpass_low=filter_low,
            bandpass_high=filter_high,
            bandpass_order=filter_order,
            downsample_factor=downsample_rate,
        )

    def find_eye_channels(self, device_spec: DeviceSpec) -> Optional[list]:
        eye_channels = []
        for channel in device_spec.channels:
            if 'F' in channel:
                eye_channels.append(channel)
        if len(eye_channels) == 0:
            eye_channels = None
        return eye_channels
    
    def get_triggers(self, session) -> tuple:
        trigger_type, trigger_timing, trigger_label = trigger_decoder(
            offset=self.static_offset,
            trigger_path=f"{session}/{TRIGGER_FILENAME}",
            exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
        )
        return trigger_type, trigger_timing, trigger_label
    
    def get_figure_handles(self, session, raw_data, channel_map) -> None:
        _, trigger_timing, trigger_label = self.get_triggers(session)
        figure_handles = visualize_erp(
                raw_data,
                channel_map,
                trigger_timing,
                trigger_label,
                self.trial_window,
                transform=self.default_transform,
                plot_average=True,
                plot_topomaps=True,
                show=False,
            )
        return figure_handles
    
    def get_artifact_detector(self, raw_data, device_spec) -> ArtifactDetection:
        eye_channels = self.find_eye_channels(device_spec)
        artifact_detector = ArtifactDetection(
            raw_data,
            self.parameters,
            device_spec,
            eye_channels=eye_channels)
        artifact_detector.detect_artifacts()
        return artifact_detector

if __name__ == '__main__':
    pass


