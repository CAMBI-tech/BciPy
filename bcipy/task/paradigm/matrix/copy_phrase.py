"""Defines the Copy Phrase Task which uses a Matrix display"""
from psychopy import visual

from bcipy.display import InformationProperties, StimuliProperties
from bcipy.display.components.task_bar import CopyPhraseTaskBar
from bcipy.display.main import PreviewParams
from bcipy.display.paradigm.matrix.display import MatrixDisplay
from bcipy.task import TaskMode
from bcipy.task.paradigm.rsvp.copy_phrase import RSVPCopyPhraseTask
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.clock import Clock


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
        "time_fixation",
        "time_flash",
        "time_prompt",
        "trial_window",
        "font",
        "fixation_color",
        "trigger_type",
        "filter_high",
        "filter_low",
        "filter_order",
        "notch_filter_frequency",
        "down_sampling_rate",
        "prestim_length",
        "is_txt_stim",
        "lm_backspace_prob",
        "backspace_always_shown",
        "decision_threshold",
        "max_inq_len",
        "max_inq_per_series",
        "max_minutes",
        "max_selections",
        "max_incorrect",
        "min_inq_len",
        "show_feedback",
        "feedback_duration",
        "show_preview_inquiry",
        "preview_inquiry_isi",
        "preview_inquiry_key_input",
        "preview_inquiry_error_prob",
        "preview_inquiry_length",
        "preview_inquiry_progress_method",
        "spelled_letters_count",
        "stim_color",
        "matrix_stim_height",
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
        "matrix_task_height",
        "matrix_task_padding",
        "matrix_keyboard_layout",
        "matrix_rows",
        "matrix_columns",
        "matrix_width",
        "matrix_stim_pos_x",
        "matrix_stim_pos_y",
        "task_text",
        "info_pos_x",
        "info_pos_y",
        "info_color",
        "info_height",
        "info_text",
        "info_color",
        "info_height",
        "info_text",
    ]

    def init_display(self) -> MatrixDisplay:
        """Initialize the Matrix display"""
        return init_display(self.parameters, self.window,
                            self.experiment_clock, self.spelled_text)


def init_display(
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
