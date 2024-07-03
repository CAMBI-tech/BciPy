"""Defines the Copy Phrase Task which uses a Matrix display"""

from bcipy.display import InformationProperties, StimuliProperties
from bcipy.display.components.task_bar import CopyPhraseTaskBar
from bcipy.display.paradigm.matrix.display import MatrixDisplay
from bcipy.task.paradigm.rsvp.copy_phrase import RSVPCopyPhraseTask


class MatrixCopyPhraseTask(RSVPCopyPhraseTask):
    """Matrix Copy Phrase Task.

    Initializes and runs all needed code for executing a copy phrase task. A
        phrase is set in parameters and necessary objects (daq, display) are
        passed to this function.

    Parameters
    ----------
        win : object,
            display window to present visual stimuli.
        daq : object,
            data acquisition object initialized for the desired protocol
        parameters : dict,
            configuration details regarding the experiment. See parameters.json
        file_save : str,
            path location of where to save data from the session
        signal_models : list of trained signal models.
        language_model: object,
            trained language model.
        fake : boolean, optional
            boolean to indicate whether this is a fake session or not.
    Returns
    -------
        file_save : str,
            path location of where to save data from the session
    """
    name = 'Matrix Copy Phrase'
    MODE = 'Matrix'

    def init_display(self) -> MatrixDisplay:
        """Initialize the Matrix display"""
        return init_display(self.parameters, self.window,
                            self.experiment_clock, self.spelled_text)


def init_display(
        parameters, win, experiment_clock, starting_spelled_text) -> MatrixDisplay:
    """Constructs a new Matrix display"""

    info = InformationProperties(
        info_color=[parameters['info_color']],
        info_pos=[(parameters['info_pos_x'], parameters['info_pos_y'])],
        info_height=[parameters['info_height']],
        info_font=[parameters['font']],
        info_text=[parameters['info_text']],
    )

    stimuli = StimuliProperties(stim_font=parameters['font'],
                                stim_pos=(parameters['stim_pos_x'],
                                          parameters['stim_pos_y']),
                                stim_height=parameters['stim_height'],
                                is_txt_stim=parameters['is_txt_stim'],
                                prompt_time=parameters['time_prompt'])

    task_bar = CopyPhraseTaskBar(win,
                                 task_text=parameters['task_text'],
                                 spelled_text=starting_spelled_text,
                                 colors=[parameters['task_color']],
                                 font=parameters['font'],
                                 height=parameters['task_height'],
                                 padding=parameters['task_padding'])

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
        should_prompt_target=False)
