# mypy: disable-error-code="assignment,arg-type"
import subprocess
from typing import Any, Optional, List, Callable, Tuple
import logging
from pathlib import Path
import glob

from bcipy.gui.bciui import run_bciui
from matplotlib.figure import Figure

from bcipy.gui.intertask_gui import IntertaskGUI
from bcipy.gui.experiments.ExperimentField import start_experiment_field_collection_gui
from bcipy.task import Task, TaskMode, TaskData
from bcipy.helpers.triggers import trigger_decoder, TriggerType

from bcipy.acquisition import devices
from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.parameters import Parameters
from bcipy.acquisition.devices import DeviceSpec
from bcipy.helpers.load import load_raw_data
from bcipy.helpers.raw_data import RawData
from bcipy.signal.process import get_default_transform
from bcipy.helpers.report import SignalReportSection, SessionReportSection, Report, ReportSection
from bcipy.config import DEFAULT_PARAMETERS_PATH, SESSION_LOG_FILENAME, RAW_DATA_FILENAME, TRIGGER_FILENAME
from bcipy.helpers.visualization import visualize_erp
from bcipy.signal.evaluate.artifact import ArtifactDetection


logger = logging.getLogger(SESSION_LOG_FILENAME)


class CodeHookAction(Task):
    """
    Action for running generic code hooks.
    """

    name = "CodeHookAction"
    mode = TaskMode.ACTION

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

    name = "OfflineAnalysisAction"
    mode = TaskMode.ACTION

    def __init__(
            self,
            parameters: Parameters,
            data_directory: str,
            parameters_path: str = f'{DEFAULT_PARAMETERS_PATH}',
            last_task_dir: Optional[str] = None,
            alert_finished: bool = False,
            **kwargs: Any) -> None:
        super().__init__()
        self.parameters = parameters
        self.parameters_path = parameters_path
        self.alert_finished = alert_finished

        # TODO: add a feature to orchestrator to permit the user to select the last task directory or have it loaded.
        if last_task_dir:
            self.data_directory = last_task_dir
        else:
            self.data_directory = data_directory

    def execute(self) -> TaskData:
        """Execute the offline analysis.

        Note: This function is called by the orchestrator to execute the offline analysis task. Some of the
            exceptions that can be raised by this function are not recoverable and will cause the orchestrator
            to stop execution. For example, if Exception is thrown in cross_validation due to the # of folds being
            inconsistent.

        """
        logger.info("Running offline analysis action")
        try:
            cmd = f"bcipy-train --parameters {self.parameters_path}"
            if self.alert_finished:
                cmd += " --alert"
            response = subprocess.run(
                cmd,
                shell=True,
                check=True,
            )
        except Exception as e:
            logger.exception(f"Error running offline analysis: {e}")
            raise e
        return TaskData(
            save_path=self.data_directory,
            task_dict={"parameters": self.parameters_path,
                       "response": response},
        )


class IntertaskAction(Task):
    name = "IntertaskAction"
    mode = TaskMode.ACTION
    tasks: List[Task]
    current_task_index: int

    def __init__(
            self,
            parameters: Parameters,
            save_path: str,
            progress: Optional[int] = None,
            tasks: Optional[List[Task]] = None,
            exit_callback: Optional[Callable] = None,
            **kwargs: Any) -> None:
        super().__init__()
        self.save_folder = save_path
        self.parameters = parameters
        assert progress is not None and tasks is not None, "Either progress or tasks must be provided"
        self.next_task_index = progress  # progress is 1-indexed, tasks is 0-indexed so we can use the same index
        assert self.next_task_index >= 0, "Progress must be greater than 1 "
        self.tasks = tasks
        self.task_name = self.tasks[self.next_task_index].name
        self.task_names = [task.name for task in self.tasks]
        self.exit_callback = exit_callback

    def execute(self) -> TaskData:

        run_bciui(
            IntertaskGUI,
            tasks=self.task_names,
            next_task_index=self.next_task_index,
            exit_callback=self.exit_callback),

        return TaskData(
            save_path=self.save_folder,
            task_dict={
                "next_task_index": self.next_task_index,
                "tasks": self.task_names,
                "task_name": self.task_name,
            },
        )

    def alert(self):
        pass


class ExperimentFieldCollectionAction(Task):
    """
    Action for collecting experiment field data.
    """

    name = "Experiment Field Collection Action"
    mode = TaskMode.ACTION

    def __init__(
            self,
            parameters: Parameters,
            data_directory: str,
            experiment_id: str = 'default',
            **kwargs: Any) -> None:
        super().__init__()
        self.experiment_id = experiment_id
        self.save_folder = data_directory
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
    Action for generating a report after calibration Tasks.
    """

    name = "BciPy Report Action"
    mode = TaskMode.ACTION

    def __init__(
            self,
            parameters: Parameters,
            save_path: str,
            protocol_path: Optional[str] = None,
            last_task_dir: Optional[str] = None,
            trial_window: Optional[Tuple[float, float]] = None,
            **kwargs: Any) -> None:
        super().__init__()
        self.save_folder = save_path
        # Currently we assume all Tasks have the same parameters, this may change in the future.
        self.parameters = parameters
        self.protocol_path = protocol_path or ''
        self.last_task_dir = last_task_dir
        self.default_transform = None
        self.trial_window = trial_window or (0, 1.0) #TODO ask about this
        self.static_offset = self.parameters.get("static_offset", 0)
        self.report = Report(self.protocol_path)
        self.report_sections: List[ReportSection] = []
        self.all_raw_data: List[RawData] = []
        self.type_amp = None

    def execute(self) -> TaskData:
        """Excute the report generation action.

        This assumes all data were collected using the same protocol, device, and parameters.
        """
        logger.info(f"Generating report in save folder {self.save_folder}")
        # loop through all the files in the last_task_dir

        data_directories = []
        # If a protocol is given, loop over and look for any calibration directories
        try:
            if self.protocol_path:
                # Use glob to find all directories with Calibration in the name
                calibration_directories = glob.glob(
                    f"{self.protocol_path}/**/*Calibration*", recursive=True)
                for data_dir in calibration_directories:
                    path_data_dir = Path(data_dir)
                    # pull out the last directory name
                    task_name = path_data_dir.parts[-1].split('_')[0]
                    data_directories.append(path_data_dir)
                    # For each calibration directory, attempt to load the raw data
                    signal_report_section = self.create_signal_report(path_data_dir)
                    session_report = self.create_session_report(path_data_dir, task_name)
                    self.report_sections.append(session_report)
                    self.report.add(session_report)
                    self.report_sections.append(signal_report_section)
                    self.report.add(signal_report_section)
            if data_directories:
                logger.info(f"Saving report generated from: {self.protocol_path}")
            else:
                logger.info(f"No data found in {self.protocol_path}")

        except Exception as e:
            logger.exception(f"Error generating report: {e}")

        self.report.compile()
        self.report.save()
        return TaskData(
            save_path=self.save_folder,
            task_dict={},
        )

    def create_signal_report(self, data_dir: Path) -> SignalReportSection:
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

        triggers = self.get_triggers(data_dir)
        # get figure handles
        figure_handles = self.get_figure_handles(raw_data, channel_map, triggers)
        artifact_detector = self.get_artifact_detector(raw_data, device_spec, triggers)
        return SignalReportSection(figure_handles, artifact_detector)

    def create_session_report(self, data_dir, task_name) -> SessionReportSection:
        # get task name
        summary_dict = {
            "task": task_name,
            "data_location": data_dir,
            "amplifier": self.type_amp
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

    def get_figure_handles(self, raw_data, channel_map, triggers) -> List[Figure]:
        _, trigger_timing, trigger_label = triggers
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

    def get_artifact_detector(self, raw_data, device_spec, triggers) -> ArtifactDetection:
        eye_channels = self.find_eye_channels(device_spec)
        artifact_detector = ArtifactDetection(
            raw_data,
            self.parameters,
            device_spec,
            eye_channels=eye_channels,
            session_triggers=triggers)
        artifact_detector.detect_artifacts()
        return artifact_detector


if __name__ == '__main__':
    pass
