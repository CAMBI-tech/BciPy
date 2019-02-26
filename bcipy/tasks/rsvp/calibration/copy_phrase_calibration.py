from psychopy import core

from bcipy.tasks.task import Task
from bcipy.display.rsvp.rsvp_disp_modes import CopyPhraseDisplay
from bcipy.helpers.triggers import _write_triggers_from_sequence_copy_phrase
from bcipy.helpers.stimuli_generation import target_rsvp_sequence_generator, get_task_info
from bcipy.helpers.bci_task_related import (
    fake_copy_phrase_decision, alphabet, get_user_input,
    trial_complete_message, pause_calibration)


class RSVPCopyPhraseCalibrationTask(Task):
    """RSVP Copy Phrase Calibration.

    Initializes and runs all needed code for executing a copy phrase
        calibration task. A phrase is set in parameters and necessary objects
        (eeg, display) are passed to this function.
        Fake decisions are made, but the implementation should mimic
        a copy phrase session.

    Parameters
    ----------
        parameters : dict,
            configuration details regarding the experiment. See parameters.json
        daq : object,
            data acquisition object initialized for the desired protocol
        file_save : str,
            path location of where to save data from the session
        classifier : loaded pickle file,
            trained signal_model, loaded before session started
        fake : boolean, optional
            boolean to indicate whether this is a fake session or not.
    Returns
    -------
        file_save : str,
            path location of where to save data from the session
    """

    def __init__(
            self, win, daq, parameters, file_save, fake):
        super(RSVPCopyPhraseCalibrationTask, self).__init__()
        self.window = win
        self.frame_rate = self.window.getActualFrameRate()
        self.parameters = parameters
        self.daq = daq
        self.static_clock = core.StaticPeriod(screenHz=self.frame_rate)
        self.experiment_clock = core.Clock()
        self.buffer_val = parameters['task_buffer_len']
        self.alp = alphabet(parameters)
        self.rsvp = _init_copy_phrase_display_task(
            self.parameters, self.window, self.daq,
            self.static_clock, self.experiment_clock)
        self.file_save = file_save
        trigger_save_location = f"{self.file_save}/{parameters['triggers_file_name']}"
        self.session_save_location = f"{self.file_save}/{parameters['session_file_name']}"
        self.trigger_file = open(trigger_save_location, 'w')

        self.wait_screen_message = parameters['wait_screen_message']
        self.wait_screen_message_color = parameters[
            'wait_screen_message_color']

        self.num_sti = parameters['num_sti']
        self.len_sti = parameters['len_sti']
        self.timing = [parameters['time_target'],
                       parameters['time_cross'],
                       parameters['time_flash']]

        self.color = [parameters['target_letter_color'],
                      parameters['fixation_color'],
                      parameters['stimuli_color']]

        self.task_info_color = parameters['task_color']

        self.stimuli_height = parameters['sti_height']

        self.is_txt_sti = parameters['is_txt_sti']
        self.eeg_buffer = parameters['eeg_buffer_len']
        self.copy_phrase = parameters['text_task']

        self.max_seq_length = parameters['max_seq_len']
        self.fake = fake

        self.enable_breaks = parameters['enable_breaks']

    def execute(self):
        self.logger.debug('Starting Copy Phrase Calibration Task!')
        run = True

        # get the initial target letter
        target_letter = self.copy_phrase[0]
        text_task = '*'

        # check user input to make sure we should be going
        if not get_user_input(self.rsvp, self.wait_screen_message,
                              self.wait_screen_message_color,
                              first_run=True):
            run = False

        while run:

            # check user input to make sure we should be going
            if not get_user_input(self.rsvp, self.wait_screen_message,
                                  self.wait_screen_message_color):
                break

            #Take a break every number of trials defined in parameters.json
            if self.enable_breaks:
                #Get number of trials by getting the length of the currently
                #typed string
                number_of_trials = len(text_task.replace('*',''))
                pause_calibration(self.window, self.rsvp, number_of_trials,
                                  self.parameters)

            # Generate some sequences to present based on parameters
            (ele_sti, timing_sti, color_sti) = target_rsvp_sequence_generator(
                self.alp, target_letter, self.parameters,
                len_sti=self.len_sti, timing=self.timing,
                is_txt=self.is_txt_sti,
                color=self.color)

            # Get task information, seperate from stimuli to be presented
            (task_text, task_color) = get_task_info(
                self.num_sti,
                self.task_info_color)

            self.rsvp.update_task_state(text=text_task, color_list=['white'])
            self.rsvp.draw_static()
            self.window.flip()

            # update task state
            self.rsvp.stim_sequence = ele_sti[0]

            # self.rsvp.text_task = text_task
            if self.is_txt_sti:
                self.rsvp.color_list_sti = color_sti[0]
            self.rsvp.time_list_sti = timing_sti[0]

            # buffer and execute the sequence
            core.wait(self.buffer_val)
            sequence_timing = self.rsvp.do_sequence()

            # Write triggers to file
            _write_triggers_from_sequence_copy_phrase(
                sequence_timing,
                self.trigger_file,
                self.copy_phrase,
                text_task)

            # buffer
            core.wait(self.buffer_val)

            # Fake a decision!
            (target_letter, text_task, run) = fake_copy_phrase_decision(
                self.copy_phrase, target_letter, text_task)

        # update the final task state and say goodbye
        self.rsvp.update_task_state(text=text_task, color_list=['white'])
        self.rsvp.text = trial_complete_message(self.window, self.parameters)
        self.rsvp.draw_static()
        self.window.flip()

        core.wait(self.buffer_val)

        if self.daq.is_calibrated:
            _write_triggers_from_sequence_copy_phrase(
                ['offset', self.daq.offset], self.trigger_file,
                self.copy_phrase, text_task, offset=True)

        # Close this sessions trigger file and return some data
        self.trigger_file.close()

        # Wait some time before exiting so there is trailing eeg data saved
        core.wait(self.eeg_buffer)

        return self.file_save

    def name(self):
        return 'RSVP Copy Phrase Calibration Task'


def _init_copy_phrase_display_task(
        parameters, win, daq, static_clock, experiment_clock):
    rsvp = CopyPhraseDisplay(
        window=win, clock=static_clock,
        experiment_clock=experiment_clock,
        marker_writer=daq.marker_writer,
        text_info=parameters['text_text'],
        static_text_task=parameters['text_task'],
        text_task='****',
        color_info=parameters['color_text'],
        pos_info=(parameters['pos_text_x'],
                  parameters['pos_text_y']),
        height_info=parameters['txt_height'],
        font_info=parameters['font_text'],
        color_task=['white'],
        font_task=parameters['font_task'],
        height_task=parameters['height_task'],
        font_sti=parameters['font_sti'],
        pos_sti=(parameters['pos_sti_x'],
                 parameters['pos_sti_y']),
        sti_height=parameters['sti_height'],
        stim_sequence=['a'] * 10, color_list_sti=['white'] * 10,
        time_list_sti=[3] * 10,
        is_txt_sti=parameters['is_txt_sti'],
        trigger_type=parameters['trigger_type'],
        space_char=parameters['sti_space_char'])

    return rsvp
