"""Defines the Copy Phrase Task which uses a Matrix display"""
import logging
from typing import Any, List, Optional, Tuple

from psychopy import visual
from psychopy.visual import Window

from bcipy.acquisition import ClientManager
from bcipy.config import (SESSION_LOG_FILENAME, WAIT_SCREEN_MESSAGE)
from bcipy.display import (InformationProperties, StimuliProperties,
                           init_display_window)
from bcipy.display.components.task_bar import CopyPhraseTaskBar
from bcipy.display.main import PreviewParams
from bcipy.display.paradigm.matrix.display import MatrixDisplay
from bcipy.exceptions import TaskConfigurationException
from bcipy.feedback.visual.visual_feedback import VisualFeedback
from bcipy.helpers.acquisition import (LslDataServer, active_content_types,
                                       init_acquisition)
from bcipy.helpers.clock import Clock
from bcipy.helpers.copy_phrase_wrapper import CopyPhraseWrapper
from bcipy.helpers.language_model import init_language_model
from bcipy.helpers.load import choose_signal_models
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.stimuli import InquirySchedule
from bcipy.helpers.task import trial_complete_message, get_user_input
from bcipy.helpers.triggers import (Trigger, TriggerType, 
                                    convert_timing_triggers, offset_label)
from bcipy.language.main import LanguageModel
from bcipy.signal.model import SignalModel
from bcipy.task import TaskMode
from bcipy.task.paradigm.rsvp.copy_phrase import RSVPCopyPhraseTask

logger = logging.getLogger(SESSION_LOG_FILENAME)


class MatrixCopyPhraseTask(RSVPCopyPhraseTask):
    """Matrix Copy Phrase Task.

    Initializes and runs all needed code for executing a copy phrase task. A
        phrase is set in parameters and necessary objects (daq, display) are
        passed to this function.

    Parameters
    ----------
        parameters : dict,
            configuration details regarding the experiment. See parameters.json
        file_save : str,
            path location of where to save data from the session
        fake : boolean, optional
            boolean to indicate whether this is a fake session or not.
    Returns
    -------
        TaskData
    """
    name = 'Matrix Copy Phrase'
    paradigm = 'Matrix'
    mode = TaskMode.COPYPHRASE

    PARAMETERS_USED = [
        "backspace_always_shown",
        "decision_threshold",
        "down_sampling_rate",
        "feedback_duration",
        "filter_high",
        "filter_low",
        "filter_order",
        "fixation_color",
        "font",
        "info_color",
        "info_color",
        "info_height",
        "info_height",
        "info_pos_x",
        "info_pos_y",
        "info_text",
        "info_text",
        "is_txt_stim",
        "lm_backspace_prob",
        "matrix_columns",
        "matrix_keyboard_layout",
        "matrix_rows",
        "matrix_stim_height",
        "matrix_stim_pos_x",
        "matrix_stim_pos_y",
        "matrix_task_height",
        "matrix_task_padding",
        "matrix_width",
        "max_incorrect",
        "max_inq_len",
        "max_inq_per_series",
        "max_minutes",
        "max_selections",
        "min_inq_len",
        "notch_filter_frequency",
        "prestim_length",
        "preview_box_text_size",
        "preview_inquiry_error_prob",
        "preview_inquiry_isi",
        "preview_inquiry_key_input",
        "preview_inquiry_length",
        "preview_inquiry_progress_method",
        "show_feedback",
        "show_preview_inquiry",
        "spelled_letters_count",
        "stim_color",
        "stim_jitter",
        "stim_length",
        "stim_number",
        "stim_order",
        "stim_pos_x",
        "stim_pos_y",
        "stim_space_char",
        "target_color",
        "task_buffer_length",
        "task_color",
        "task_text",
        "time_fixation",
        "time_flash",
        "time_prompt",
        "trial_window",
        "trigger_type",
    ]

    def __init__(self, parameters: Parameters, file_save: str, fake: bool = False):
        super(MatrixCopyPhraseTask, self).__init__(parameters, file_save, fake)
        self.matrix = self.init_display()
        
    def init_display(self) -> MatrixDisplay:
        """Initialize the Matrix display"""
        return _init_matrix_display(self.parameters, 
                            self.window, 
                            self.experiment_clock, 
                            self.spelled_text,
        )

    def setup(
            self,
            parameters: Parameters,
            data_save_location: str,
            fake: bool = False) -> Tuple[ClientManager, List[LslDataServer], Window]:
        # Initialize Acquisition
        daq, servers = init_acquisition(
            parameters, data_save_location, server=fake)

        # Initialize Display
        display = init_display_window(parameters)
        self.initalized = True

        return daq, servers, display
    
    def get_language_model(self) -> LanguageModel:
        return init_language_model(self.parameters)

    def get_signal_models(self) -> Optional[List[SignalModel]]:
        if not self.fake:
            try:
                signal_models = choose_signal_models(
                    active_content_types(self.parameters['acq_mode']))
                assert signal_models, "No signal models selected"
            except Exception as error:
                logger.exception(f'Cannot load signal models. Exiting. {error}')
                raise error
            return signal_models
        return []
    
    def cleanup(self):
        return super().cleanup()

    def save_session_data(self):
        return super().save_session_data()

    def user_wants_to_continue(self) -> bool:
        """Check if user wants to continue or terminate.

        Returns
        -------
        - `True` to continue
        - `False` to finish the task.
        """
        should_continue = get_user_input(
            self.matrix,
            WAIT_SCREEN_MESSAGE,
            self.parameters["stim_color"],
            first_run=self.first_run,
        )
        if not should_continue:
            logger.info("User wants to exit.")
        return should_continue
    
    def present_inquiry(
        self, inquiry_schedule: InquirySchedule
    ) -> Tuple[List[Tuple[str, float]], bool]:
        """Present the given inquiry and return the trigger timing info.

        Parameters
        ----------
        - inquiry_schedule : schedule for next sequences of stimuli to present.
        Currently, only the first list of stims in the schedule is used.

        Returns
        -------
        Tuple of `(stim_times, proceed)`

        - stim_times : list of tuples representing the stimulus and time that
        it was presented relative to the experiment clock. Non-stim triggers
        may be also be included in the list ('calibration', etc).
        - proceed : indicates whether to proceed with evaluating eeg evidence.
        a value of False indicates that the inquiry was previewed but not
        presented in sequence.
        """
        # Update task state and reset the static
        self.matrix.update_task_bar(self.spelled_text)
        self.matrix.draw_static()
        self.window.flip()

        self.wait()

        # Setup the new stimuli
        self.matrix.schedule_to(
            stimuli=inquiry_schedule.stimuli[0],
            timing=inquiry_schedule.durations[0],
            colors=(
                inquiry_schedule.colors[0] if self.parameters["is_txt_stim"] else None
            ),
        )

        stim_times = self.matrix.do_inquiry()
        proceed = True  # `proceed` is always True for matrix task, since it does't have inquiry preview
        
        return stim_times, proceed
    
    def exit_display(self) -> None:
        """Close the UI and cleanup."""
        # Update task state and reset the static
        self.matrix.update_task_bar(text=self.spelled_text)

        # Say Goodbye!
        self.matrix.info_text = trial_complete_message(self.window, self.parameters)
        self.matrix.draw_static()
        self.window.flip()

        # Give the system time to process
        self.wait()
    
    def write_offset_trigger(self) -> None:
        """Append the offset to the end of the triggers file."""
        # To help support future refactoring or use of lsl timestamps only
        # we write only the sample offset here.
        triggers = []
        for content_type, client in self.daq.clients_by_type.items():
            label = offset_label(content_type.name, prefix="daq_sample_offset")
            time = client.offset(self.matrix.first_stim_time)
            triggers.append(Trigger(label, TriggerType.SYSTEM, time))

        self.trigger_handler.add_triggers(triggers)
        self.trigger_handler.close()
    
    def write_trigger_data(
        self, stim_times: List[Tuple[str, float]], target_stimuli: str
    ) -> None:
        """Save trigger data to disk.

        Parameters
        ----------
        - stim_times : list of (stim, clock_time) tuples
        - target_stimuli : current target stimuli
        """

        if self.first_run:
            # offset will factor in true offset and time relative from beginning
            offset_triggers = []
            for content_type, client in self.daq.clients_by_type.items():
                label = offset_label(content_type.name)
                time = (
                    client.offset(self.matrix.first_stim_time) - self.matrix.first_stim_time
                )
                offset_triggers.append(Trigger(label, TriggerType.OFFSET, time))
            self.trigger_handler.add_triggers(offset_triggers)

        triggers = convert_timing_triggers(
            stim_times, target_stimuli, self.trigger_type
        )
        self.trigger_handler.add_triggers(triggers)


def _init_matrix_display(
        parameters: Parameters,
        win: visual.Window,
        experiment_clock: Clock,
        starting_spelled_text: str) -> MatrixDisplay:
    """Constructs a new Matrix display"""

    info = InformationProperties(
        info_color=[parameters['info_color']],
        info_pos=[(parameters['info_pos_x'], parameters['info_pos_y'])],
        info_height=[parameters['info_height']],
        info_font=[parameters['font']],
        info_text=[parameters['info_text']],
    )

    stimuli = StimuliProperties(stim_font=parameters['font'],
                                stim_pos=(parameters['matrix_stim_pos_x'],
                                          parameters['matrix_stim_pos_y']),
                                stim_height=parameters['matrix_stim_height'],
                                is_txt_stim=parameters['is_txt_stim'],
                                prompt_time=parameters['time_prompt'],
                                layout=parameters['matrix_keyboard_layout'])

    task_bar = CopyPhraseTaskBar(win,
                                 task_text=parameters['task_text'],
                                 spelled_text=starting_spelled_text,
                                 colors=[parameters['task_color']],
                                 font=parameters['font'],
                                 height=parameters['matrix_task_height'],
                                 padding=parameters['matrix_task_padding'])

    return MatrixDisplay(
        win,
        experiment_clock,
        stimuli,
        task_bar,
        info,
        rows=parameters['matrix_rows'],
        columns=parameters['matrix_columns'],
        width_pct=parameters['matrix_width'],
        height_pct=1 - (2 * task_bar.height_pct),
        trigger_type=parameters['trigger_type'],
        should_prompt_target=False,
        preview_config=parameters.instantiate(PreviewParams))


