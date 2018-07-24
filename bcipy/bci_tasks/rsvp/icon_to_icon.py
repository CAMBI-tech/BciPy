from psychopy import core

from bcipy.bci_tasks.task import Task

from bcipy.display.rsvp.rsvp_disp_modes import IconToIconDisplay

from bcipy.helpers.stimuli_generation import generate_icon_match_images

from bcipy.helpers.triggers import _write_triggers_from_sequence_copy_phrase

from bcipy.helpers.eeg_model_related import CopyPhraseWrapper

from bcipy.helpers.save import _save_session_related_data

from bcipy.feedback.visual.visual_feedback import VisualFeedback

from bcipy.helpers.bci_task_related import (
    fake_copy_phrase_decision, alphabet, process_data_for_decision,
    trial_complete_message, get_user_input)
    
import glob

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
            self, win, daq, parameters, file_save, classifier, lmodel, fake):

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

        self.eeg_buffer = parameters['eeg_buffer_len']

        self.max_seq_length = parameters['max_seq_len']
        self.fake = fake
        self.lmodel = lmodel
        self.classifier = classifier

        self.image_path = parameters['path_to_presentation_images']
        self.task_height = parameters['height_task']

        self.is_txt_sti = False
        
        self.min_num_seq = parameters['min_seq_len']

    def execute(self):
        image_array, timing_array = generate_icon_match_images(self.len_sti,
                                                           self.image_path,
                                                           self.num_sti,
                                                           self.timing)
                     
        #Get all png images in image path
        alp_image_array = glob.glob(self.image_path + '*.png')

        #Remove plus image from array
        for image in alp_image_array:
            if image.endswith('PLUS.png'):
                alp_image_array.remove(image)
        for image in alp_image_array:
            alp_image_array[alp_image_array.index(image)] = image.replace('.png', '')
                
        self.alp = alp_image_array
        
        # Try Initializing Copy Phrase Wrapper:
        #       (sig_pro, decision maker, signal_model)
        try:
            copy_phrase_task = CopyPhraseWrapper(self.min_num_seq, self.max_seq_length, signal_model=self.classifier, fs=self.daq.device_info.fs,
                                                 k=2, alp=self.alp, task_list=['unnecessary_string', 'unnecessary_string'],
                                                 lmodel=self.lmodel,
                                                 is_txt_sti=self.is_txt_sti,
                                                 device_name=self.daq.device_info.name,
                                                 device_channels=self.daq.device_info.channels)
        except Exception as e:
            print("Error initializing Copy Phrase Task")
            raise e
            
        run = True
        
        # Init session data and save before beginning
        data = {
            'session': self.file_save,
            'session_type': 'Icon to Icon Matching',
            'paradigm': 'RSVP',
            'epochs': {},
            'total_time_spent': self.experiment_clock.getTime(),
            'total_number_epochs': 0,
        }
        
        # Save session data
        _save_session_related_data(self.session_save_location, data)

        # Check user input to make sure we should be going
        if not get_user_input(self.rsvp, self.wait_screen_message,
                              self.wait_screen_message_color,
                              first_run=True):
            run = False

        while run:
            for each_trial in range(len(image_array)):
                # check user input to make sure we should be going
                if not get_user_input(self.rsvp, self.wait_screen_message,
                                      self.wait_screen_message_color):
                    break
                
                self.rsvp.sti.height = self.stimuli_height

                self.rsvp.stim_sequence = image_array[each_trial]
                self.rsvp.time_list_sti = timing_array
                core.wait(self.buffer_val)
                
                self.rsvp.update_task_state(self.rsvp.stim_sequence[0], self.task_height, 'yellow', self.rsvp.win.size)

                # Do the sequence
                sequence_timing = self.rsvp.do_sequence()

                # Wait for a time
                core.wait(self.buffer_val)
                
                # reshape the data and triggers as needed for later modules
                raw_data, triggers, target_info = \
                    process_data_for_decision(sequence_timing, self.daq)
                    
                
                if self.fake:
                    correct_decision = True
                else:
                    correct_decision, sti = \
                        copy_phrase_task.evaluate_sequence(raw_data, triggers,
                                                           target_info)
                                                       
                
                if correct_decision:
                    message_color = 'green'
                else:
                    message_color = 'red'
                    
                #Display feedback about whether decision was correct
                visual_feedback = VisualFeedback(
                display=self.rsvp.win, parameters=self.parameters, clock=self.experiment_clock)
                stimulus = self.rsvp.stim_sequence[0]
                visual_feedback.message_color = message_color
                visual_feedback.administer(stimulus, compare_assertion=None, message='Decision:')

            run = False

        # Say Goodbye!
        self.rsvp.text = trial_complete_message(self.window, self.parameters)
        self.rsvp.draw_static()
        self.window.flip()

        # Give the system time to process
        core.wait(self.buffer_val)

        # Close this sessions trigger file and return some data
        self.trigger_file.close()

        # Wait some time before exiting so there is trailing eeg data saved
        core.wait(self.eeg_buffer)

        return self.file_save


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
        color_task=['black'],
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
        is_txt_sti=False,
        trigger_type=parameters['trigger_type'])

    return rsvp
