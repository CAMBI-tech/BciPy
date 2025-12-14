# mypy: disable-error-code="assignment,arg-type"
"""Task actions module for BCI tasks.

This module provides various task actions that can be executed as part of a BCI
experiment, including code hooks, offline analysis, intertask management, and
report generation.
"""

import glob
import logging
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from matplotlib.figure import Figure

from bcipy.acquisition import devices
from bcipy.acquisition.devices import DeviceSpec
from bcipy.config import (RAW_DATA_FILENAME, SESSION_LOG_FILENAME,
                          TRIGGER_FILENAME)
from bcipy.core.parameters import Parameters
from bcipy.core.raw_data import RawData
from bcipy.core.report import (Report, ReportSection, SessionReportSection,
                               SignalReportSection)
from bcipy.core.triggers import TriggerType, trigger_decoder
from bcipy.gui.bciui import run_bciui
from bcipy.gui.experiments.ExperimentField import \
    start_experiment_field_collection_gui
from bcipy.gui.file_dialog import ask_directory
from bcipy.gui.intertask_gui import IntertaskGUI
from bcipy.helpers.acquisition import analysis_channels
from bcipy.helpers.visualization import visualize_erp
from bcipy.io.load import load_raw_data
from bcipy.signal.evaluate.artifact import ArtifactDetection
from bcipy.signal.process import get_default_transform
from bcipy.task import Task, TaskData, TaskMode

logger = logging.getLogger(SESSION_LOG_FILENAME)


class CodeHookAction(Task):
    """Action for running generic code hooks.

    Attributes:
        name: Name of the task.
        mode: Task execution mode.
        code_hook: Code to execute.
        subprocess: Whether to run in a subprocess.
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
        """Initialize the code hook action.

        Args:
            parameters: Task parameters.
            data_directory: Directory for data storage.
            code_hook: Code to execute.
            subprocess: Whether to run in a subprocess.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.code_hook = code_hook
        self.subprocess = subprocess

    def execute(self) -> TaskData:
        """Execute the code hook.

        Returns:
            TaskData: Empty task data.
        """
        if self.code_hook:
            if self.subprocess:
                subprocess.Popen(self.code_hook, shell=True)
            else:
                subprocess.run(self.code_hook, shell=True)
        return TaskData()


class OfflineAnalysisAction(Task):
    """Action for running offline analysis.

    Attributes:
        name: Name of the task.
        mode: Task execution mode.
        parameters: Task parameters.
        parameters_path: Path to parameters file.
        data_directory: Directory containing data to analyze.
        alert_finished: Whether to alert when analysis completes.
    """

    name = "OfflineAnalysisAction"
    mode = TaskMode.ACTION

    def __init__(
            self,
            parameters: Parameters,
            data_directory: str,
            parameters_path: str,
            last_task_dir: Optional[str] = None,
            alert_finished: bool = False,
            **kwargs: Any) -> None:
        """Initialize the offline analysis action.

        Args:
            parameters: Task parameters.
            data_directory: Directory containing data to analyze.
            parameters_path: Path to parameters file.
            last_task_dir: Directory of last executed task.
            alert_finished: Whether to alert when analysis completes.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.parameters = parameters
        self.parameters_path = parameters_path
        self.alert_finished = alert_finished

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

        Returns:
            TaskData: Contains analysis results and parameters.

        Raises:
            Exception: If offline analysis fails.
        """
        logger.info("Running offline analysis action")
        try:
            # Note: The subprocess.run function will cause a segmentation fault if visualization and alerting are
            # enabled. This is because the MNE library.
            cmd = f'bcipy-train -p "{self.parameters_path}"'
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
    """Action for managing transitions between tasks.

    Attributes:
        name: Name of the task.
        mode: Task execution mode.
        tasks: List of tasks to manage.
        current_task_index: Index of current task.
        save_folder: Directory for saving task data.
        parameters: Task parameters.
        next_task_index: Index of next task to execute.
        task_name: Name of current task.
        task_names: List of task names.
        exit_callback: Function to call on exit.
    """

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
        """Initialize the intertask action.

        Args:
            parameters: Task parameters.
            save_path: Directory for saving task data.
            progress: Current progress (1-indexed).
            tasks: List of tasks to manage.
            exit_callback: Function to call on exit.
            **kwargs: Additional keyword arguments.

        Raises:
            AssertionError: If progress or tasks is None, or if progress < 0.
        """
        super().__init__()
        self.save_folder = save_path
        self.parameters = parameters
        assert progress is not None and tasks is not None, "Either progress or tasks must be provided"
        # progress is 1-indexed, tasks is 0-indexed so we can use the same index
        self.next_task_index = progress
        assert self.next_task_index >= 0, "Progress must be greater than 1 "
        self.tasks = tasks
        self.task_name = self.tasks[self.next_task_index].name
        self.task_names = [task.name for task in self.tasks]
        self.exit_callback = exit_callback

    def execute(self) -> TaskData:
        """Execute the intertask action.

        Returns:
            TaskData: Contains task state information.
        """
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
        """Handle alerts (not implemented)."""
        pass


class ExperimentFieldCollectionAction(Task):
    """Action for collecting experiment field data.

    Attributes:
        name: Name of the task.
        mode: Task execution mode.
        experiment_id: Identifier for the experiment.
        save_folder: Directory for saving collected data.
        parameters: Task parameters.
    """

    name = "ExperimentFieldCollectionAction"
    mode = TaskMode.ACTION

    def __init__(
            self,
            parameters: Parameters,
            data_directory: str,
            experiment_id: str = 'default',
            **kwargs: Any) -> None:
        """Initialize the experiment field collection action.

        Args:
            parameters: Task parameters.
            data_directory: Directory for saving collected data.
            experiment_id: Identifier for the experiment.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.experiment_id = experiment_id
        self.save_folder = data_directory
        self.parameters = parameters

    def execute(self) -> TaskData:
        """Execute the experiment field collection.

        Returns:
            TaskData: Contains experiment metadata.
        """
        logger.info(
            f"Collecting experiment field data for experiment {self.experiment_id} in save folder {self.save_folder}"
        )
        start_experiment_field_collection_gui(
            self.experiment_id, self.save_folder)
        return TaskData(
            save_path=self.save_folder,
            task_dict={
                "experiment_id": self.experiment_id,
            },
        )


class BciPyCalibrationReportAction(Task):
    """Action for generating a report after calibration Tasks.

    Attributes:
        name: Name of the task.
        mode: Task execution mode.
        parameters: Task parameters.
        save_folder: Directory for saving reports.
        protocol_path: Path to protocol file.
        last_task_dir: Directory of last executed task.
        trial_window: Time window for trial analysis.
        report: Report instance.
        report_sections: List of report sections.
        all_raw_data: List of raw data.
        default_transform: Signal transformation function.
        type_amp: Amplifier type.
        static_offset: Static offset value.
    """

    name = "BciPyReportAction"
    mode = TaskMode.ACTION

    def __init__(
            self,
            parameters: Parameters,
            save_path: str,
            protocol_path: Optional[str] = None,
            last_task_dir: Optional[str] = None,
            trial_window: Optional[Tuple[float, float]] = None,
            **kwargs: Any) -> None:
        """Initialize the calibration report action.

        Args:
            parameters: Task parameters.
            save_path: Directory for saving reports.
            protocol_path: Path to protocol file.
            last_task_dir: Directory of last executed task.
            trial_window: Time window for trial analysis.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.save_folder = save_path
        # Currently we assume all Tasks have the same parameters, this may change in the future.
        self.parameters = parameters

        if not protocol_path:
            protocol_path = ask_directory(
                prompt="Select BciPy protocol directory with calibration data...",
                strict=True)
        self.protocol_path = protocol_path
        self.last_task_dir = last_task_dir
        self.trial_window = (-0.2, 1.0)
        self.report = Report(self.protocol_path)
        self.report_sections: List[ReportSection] = []
        self.all_raw_data: List[RawData] = []

        # These are pulled off the device spec / set from parameters
        self.default_transform = None
        self.type_amp = None
        self.static_offset = None

    def execute(self) -> TaskData:
        """Execute the report generation action.

        This assumes all data were collected using the same protocol, device, and parameters.

        Returns:
            TaskData: Contains report data and metadata.
        """
        logger.info(f"Generating report in save folder {self.save_folder}")
        # loop through all the files in the last_task_dir

        data_directories = []
        # If a protocol is given, loop over and look for any calibration directories
        try:
            if self.protocol_path:
                # Use glob to find all directories with Calibration in the name
                calibration_directories = glob.glob(
                    f"{self.protocol_path}/**/*Calibration*",
                    recursive=True)
                for data_dir in calibration_directories:
                    path_data_dir = Path(data_dir)
                    # pull out the last directory name
                    task_name = path_data_dir.parts[-1].split('_')[0]
                    data_directories.append(path_data_dir)
                    # For each calibration directory, attempt to load the raw data
                    signal_report_section = self.create_signal_report(
                        path_data_dir)
                    session_report = self.create_session_report(
                        path_data_dir, task_name)
                    self.report_sections.append(session_report)
                    self.report.add(session_report)
                    self.report_sections.append(signal_report_section)
                    self.report.add(signal_report_section)
            if data_directories:
                logger.info(
                    f"Saving report generated from: {self.protocol_path}")
            else:
                logger.info(f"No data found in {self.protocol_path}")

        except Exception as e:
            logger.exception(f"Error generating report: {e}")

        self.report.compile()
        self.report.save()

        return TaskData(
            save_path=self.save_folder,
            task_dict={
                'reports': len(calibration_directories),
            },
        )

    def create_signal_report(self, data_dir: Path) -> SignalReportSection:
        """Create a report section for signal quality metrics.

        Args:
            data_dir: Directory containing signal data.

        Returns:
            SignalReportSection: Report section containing signal metrics.
        """
        raw_data = load_raw_data(Path(data_dir, f'{RAW_DATA_FILENAME}.csv'))
        if not self.type_amp:
            self.type_amp = raw_data.daq_type
        channels = raw_data.channels
        sample_rate = raw_data.sample_rate
        device_spec = devices.preconfigured_device(raw_data.daq_type)
        self.static_offset = device_spec.static_offset
        channel_map = analysis_channels(channels, device_spec)

        logger.info(
            f"Creating signal section for {data_dir}. \n"
            f"Channels: {channels}, Sample Rate: {sample_rate}, "
            f"Device: {self.type_amp}, Static Offset: {self.static_offset}")

        # Set the default transform if not already set
        if not self.default_transform:
            self.set_default_transform(sample_rate)

        triggers = self.get_triggers(data_dir)
        # get figure handles
        figure_handles = self.get_figure_handles(
            raw_data, channel_map, triggers)
        artifact_detector = self.get_artifact_detector(
            raw_data, device_spec, triggers)
        return SignalReportSection(figure_handles, artifact_detector)

    def create_session_report(self, data_dir: Path, task_name: str) -> SessionReportSection:
        """Create a report section for session information.

        Args:
            data_dir: Directory containing session data.
            task_name: Name of the task.

        Returns:
            SessionReportSection: Report section containing session info.
        """
        # get task name
        summary_dict = {
            "task": task_name,
            "data_location": data_dir,
            "amplifier": self.type_amp
        }
        signal_model_metrics = self.get_signal_model_metrics(data_dir)
        summary_dict.update(signal_model_metrics)

        return SessionReportSection(summary_dict)

    def get_signal_model_metrics(self, data_directory: Path) -> Dict[str, Any]:
        """Get the signal model metrics from the session folder.

        In the future, the model will save a ModelMetrics with the pkl file.
        For now, we just look for the pkl file and extract the AUC from the filename.

        Args:
            data_directory: Directory containing model data.

        Returns:
            Dict[str, Any]: Dictionary of model metrics.
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
        """Set the default signal transformation function.

        Args:
            sample_rate: Sampling rate of the signal.
        """
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

    def find_eye_channels(self, device_spec: DeviceSpec) -> Optional[List[str]]:
        """Find eye-tracking channels in the device specification.

        Args:
            device_spec: Device specification.

        Returns:
            Optional[List[str]]: List of eye channel names if found.
        """
        eye_channels = []
        for channel in device_spec.channels:
            if 'F' in channel:
                eye_channels.append(channel)
        if len(eye_channels) == 0:
            eye_channels = None
        return eye_channels

    def get_triggers(self, session: str) -> Tuple[List[Any], List[float], List[str]]:
        """Get triggers from the session data.

        Args:
            session: Path to session directory.

        Returns:
            Tuple[List[Any], List[float], List[str]]: Trigger type, timing, and labels.
        """
        trigger_type, trigger_timing, trigger_label = trigger_decoder(
            offset=self.static_offset,
            trigger_path=f"{session}/{TRIGGER_FILENAME}",
            exclusion=[
                TriggerType.PREVIEW,
                TriggerType.EVENT,
                TriggerType.FIXATION],
            device_type='EEG'
        )
        return trigger_type, trigger_timing, trigger_label

    def get_figure_handles(self, raw_data: RawData, channel_map: List[str],
                           triggers: Tuple[TriggerType, List[float], List[str]]) -> List[Figure]:
        """Generate figures for the report.

        Args:
            raw_data: Raw signal data.
            channel_map: List of channel names.
            triggers: Tuple of trigger type, timing, and labels.

        Returns:
            List[Figure]: List of generated figures.
        """
        trigger_type, trigger_timing, _ = triggers
        figure_handles = visualize_erp(
            raw_data,
            channel_map,
            trigger_timing,
            trigger_type,
            self.trial_window,
            transform=self.default_transform,
            plot_average=True,
            plot_topomaps=True,
            show=False,
        )
        return figure_handles

    def get_artifact_detector(self, raw_data: RawData, device_spec: DeviceSpec,
                              triggers: Tuple[TriggerType, List[float], List[str]]) -> ArtifactDetection:
        """Create an artifact detector for the signal data.

        Args:
            raw_data: Raw signal data.
            device_spec: Device specification.
            triggers: Tuple of trigger type, timing, and labels.

        Returns:
            ArtifactDetection: Configured artifact detector.
        """
        eye_channels = self.find_eye_channels(device_spec)
        artifact_detector = ArtifactDetection(
            raw_data,
            self.parameters,
            device_spec,
            eye_channels=eye_channels,
            session_triggers=triggers)
        artifact_detector.detect_artifacts()
        return artifact_detector
