"""Defines the Copy Phrase Task which uses a Matrix display"""

from bcipy.task.paradigm.rsvp.copy_phrase import RSVPCopyPhraseTask
from bcipy.display import Display, InformationProperties, TaskDisplayProperties, StimuliProperties
from bcipy.display.paradigm.matrix.mode.copy_phrase import MatrixCopyPhraseDisplay


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
        signal_model : loaded pickle file,
            trained signal model.
        language_model: object,
            trained language model.
        fake : boolean, optional
            boolean to indicate whether this is a fake session or not.
    Returns
    -------
        file_save : str,
            path location of where to save data from the session
    """
    TASK_NAME = 'Matrix Copy Phrase Task'
    MODE = 'Matrix'

    def init_display(self) -> Display:
        """Initialize the Matrix display"""
        return init_display(self.parameters, self.window,
                            self.experiment_clock, self.spelled_text)


def init_display(parameters, win, experiment_clock, starting_spelled_text):
    """Constructs a new Matrix display"""
    # preview_inquiry = PreviewInquiryProperties(
    #     preview_only=parameters['preview_only'],
    #     preview_inquiry_length=parameters['preview_inquiry_length'],
    #     preview_inquiry_key_input=parameters['preview_inquiry_key_input'],
    #     preview_inquiry_progress_method=parameters[
    #         'preview_inquiry_progress_method'],
    #     preview_inquiry_isi=parameters['preview_inquiry_isi'])

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

    task_display = TaskDisplayProperties(
        task_color=[parameters['task_color']],
        task_pos=None,
        task_font=parameters['font'],
        task_height=parameters['task_height'],
        task_text=parameters['task_text'])

    return MatrixCopyPhraseDisplay(
        win,
        experiment_clock,
        stimuli,
        task_display,
        info,
        trigger_type=parameters['trigger_type'],
        starting_spelled_text=starting_spelled_text)
