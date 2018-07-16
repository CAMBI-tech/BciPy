from psychopy import core

from bcipy.bci_tasks.task import Task

from bcipy.display.rsvp.rsvp_disp_modes import IconToIconDisplay

from bcipy.helpers.stimuli_generation import generate_icon_match_images

from bcipy.helpers.bci_task_related import (
    fake_copy_phrase_decision, alphabet, process_data_for_decision,
    trial_complete_message, get_user_input)

class RSVPIconToIconTask(Task):
    """RSVP Icon to Icon Matching Task.

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

        self.window = win
        self.frame_rate = self.window.getActualFrameRate()
        self.parameters = parameters
        self.daq = daq
        self.static_clock = core.StaticPeriod(screenHz=self.frame_rate)
        self.experiment_clock = core.Clock()
        self.buffer_val = parameters['task_buffer_len']
        self.alp = alphabet(parameters)
        self.rsvp = _init_icon_to_icon_display_task(
            self.parameters, self.window, self.daq,
            self.static_clock, self.experiment_clock)
        self.file_save = file_save

        trigger_save_location = f"{self.file_save}/{parameters['triggers_file_name']}"
        self.trigger_file = open(trigger_save_location, 'w')
        self.session_save_location = f"{self.file_save}/{parameters['session_file_name']}"

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

        self.max_seq_length = parameters['max_seq_len']
        self.fake = fake

        self.image_path = parameters['path_to_presentation_images']

    def execute(self):
        run = True

        # Check user input to make sure we should be going
        if not get_user_input(self.rsvp, self.wait_screen_message,
                              self.wait_screen_message_color,
                              first_run=True):
            run = False

        image_array = generate_icon_match_images(self.len_sti, self.image_path,
                                                 self.num_sti)

        while run:
            # check user input to make sure we should be going
            if not get_user_input(self.rsvp, self.wait_screen_message,
                                  self.wait_screen_message_color):
                break


def _init_icon_to_icon_display_task(
        parameters, win, daq, static_clock, experiment_clock):
    rsvp = IconToIconDisplay(
        window=win, clock=static_clock,
        experiment_clock=experiment_clock,
        marker_writer=daq.marker_writer,
        text_info=parameters['text_text'],
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
        tr_pos_bg=(parameters['tr_pos_bg_x'],
                   parameters['tr_pos_bg_y']),
        bl_pos_bg=(parameters['bl_pos_bg_x'],
                   parameters['bl_pos_bg_y']),
        size_domain_bg=parameters['size_domain_bg'],
        color_bg_txt=parameters['color_bg_txt'],
        font_bg_txt=parameters['font_bg_txt'],
        color_bar_bg=parameters['color_bar_bg'],
        is_txt_sti=parameters['is_txt_sti'],
        trigger_type=parameters['trigger_type'])

    return rsvp
