from typing import List, NamedTuple, TextIO, Tuple

import numpy as np
from psychopy import core

from bcipy.display.rsvp import (InformationProperties,
                                PreviewInquiryProperties, StimuliProperties,
                                TaskDisplayProperties)
from bcipy.display.rsvp.mode.copy_phrase import CopyPhraseDisplay
from bcipy.feedback.visual.visual_feedback import VisualFeedback
from bcipy.helpers.copy_phrase_wrapper import CopyPhraseWrapper, EvidenceName
from bcipy.helpers.save import _save_session_related_data
from bcipy.helpers.stimuli import InquirySchedule
from bcipy.helpers.task import (BACKSPACE_CHAR, alphabet, construct_triggers,
                                fake_copy_phrase_decision, get_user_input,
                                process_data_for_decision, target_info,
                                trial_complete_message)
from bcipy.helpers.triggers import _write_triggers_from_inquiry_copy_phrase
from bcipy.signal.model.inquiry_preview import compute_probs_after_preview
from bcipy.tasks.rsvp.main_frame import EvidenceFusion
from bcipy.tasks.session_data import Inquiry, Session
from bcipy.tasks.task import Task


class Decision(NamedTuple):
    """Represents the result of evaluating evidence.
    
    Attrs
    -----
    - decision_made : whether or a not there was a commitment to a letter.
    - selection : selected letter; will be an empty string if there was no
    commitment.
    - spelled_text : spelled text resulting from the decision.
    - new_inq_schedule : the next inquiry to present if there was not a
    decision.
    """
    decision_made: bool
    selection: str
    spelled_text: str
    new_inq_schedule: InquirySchedule


class RSVPCopyPhraseTask(Task):
    """RSVP Copy Phrase Task.

    Initializes and runs all needed code for executing a copy phrase task. A
        phrase is set in parameters and necessary objects (daq, display) are
        passed to this function. Certain Wrappers and Task Specific objects are
        executed here.

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

    TASK_NAME = 'RSVP Copy Phrase Task'
    PARAMETERS_USED = [
        'backspace_always_shown', 'decision_threshold', 'down_sampling_rate',
        'eeg_buffer_len', 'feedback_flash_time', 'feedback_font',
        'feedback_line_width', 'feedback_message_color', 'feedback_pos_x',
        'feedback_pos_y', 'feedback_stim_height', 'feedback_stim_width',
        'filter_high', 'filter_low', 'filter_order', 'fixation_color',
        'info_color', 'info_font', 'info_height', 'info_text', 'is_txt_stim',
        'lm_backspace_prob', 'max_inq_len', 'max_inq_per_trial', 'max_minutes',
        'min_inq_len', 'notch_filter_frequency', 'preview_inquiry_isi',
        'preview_inquiry_key_input', 'preview_inquiry_length',
        'preview_inquiry_progress_method', 'session_file_name',
        'show_feedback', 'show_preview_inquiry', 'spelled_letters_count',
        'static_trigger_offset', 'stim_color', 'stim_font', 'stim_height',
        'stim_length', 'stim_number', 'stim_pos_x', 'stim_pos_y',
        'stim_space_char', 'target_color', 'task_buffer_len', 'task_color',
        'task_font', 'task_height', 'task_text', 'text_pos_x', 'text_pos_y',
        'time_cross', 'time_flash', 'time_target', 'trial_complete_message',
        'trial_complete_message_color', 'trial_length', 'trigger_file_name',
        'trigger_type', 'wait_screen_message', 'wait_screen_message_color'
    ]

    def __init__(self, win, daq, parameters, file_save, signal_model,
                 language_model, fake):
        super(RSVPCopyPhraseTask, self).__init__()

        self.window = win
        self.parameters = parameters
        self.daq = daq
        self.static_clock = core.StaticPeriod(
            screenHz=self.window.getActualFrameRate())
        self.experiment_clock = core.Clock()

        self.alp = alphabet(parameters)
        self.rsvp = _init_copy_phrase_display(self.parameters, self.window,
                                              self.daq, self.static_clock,
                                              self.experiment_clock)
        self.file_save = file_save

        self.trigger_save_location = f"{self.file_save}/{parameters['trigger_file_name']}"
        self.session_save_location = f"{self.file_save}/{parameters['session_file_name']}"

        self.copy_phrase = parameters['task_text']

        self.fake = fake
        self.language_model = language_model
        self.signal_model = signal_model

        # TODO: add a parameter for button_press_error_prob.
        self.button_press_error_prob = 0.05
        self.evidence_names = [EvidenceName.LM, EvidenceName.ERP]
        if self.parameters['show_preview_inquiry']:
            self.evidence_names.append(EvidenceName.BTN)

    def setup(self) -> None:
        """Initialize/reset parameters used in the execute run loop."""

        self.spelled_text = str(
            self.copy_phrase[0:self.starting_spelled_letters()])
        self.last_selection = ''
        self.inq_counter = 0
        self.session = Session(save_location=self.file_save,
                               task='Copy Phrase',
                               mode='RSVP')
        self.write_session_data()

        self.init_copy_phrase_task()
        self.current_inquiry = self.next_inquiry()

    def starting_spelled_letters(self) -> int:
        """Number of letters already spelled at the start of the task."""
        spelled_letters_count = self.parameters['spelled_letters_count']
        if spelled_letters_count > len(self.copy_phrase):
            self.logger.debug('Already spelled letters exceeds phrase length.')
            spelled_letters_count = 0
        return spelled_letters_count

    def next_inquiry(self) -> InquirySchedule:
        """Generate an InquirySchedule for spelling the next letter."""
        if self.copy_phrase_task:
            _, inquiry_schedule = self.copy_phrase_task.initialize_series()
            return inquiry_schedule
        return None

    def init_copy_phrase_task(self) -> CopyPhraseWrapper:
        """Initialize the CopyPhraseWrapper.

        Returns:
        --------
        initialized CopyPhraseWrapper
        """

        self.copy_phrase_task = _init_copy_phrase_wrapper(
            self.parameters['min_inq_len'],
            self.parameters['max_inq_per_trial'],
            signal_model=self.signal_model,
            fs=self.daq.device_info.fs,
            k=self.parameters['down_sampling_rate'],
            alp=self.alp,
            evidence_names=self.evidence_names,
            task_list=[(str(self.copy_phrase), self.spelled_text)],
            lmodel=self.language_model,
            is_txt_stim=self.parameters['is_txt_stim'],
            device_name=self.daq.device_info.name,
            device_channels=self.daq.device_info.channels,
            stimuli_timing=[
                self.parameters['time_cross'], self.parameters['time_flash']
            ],
            decision_threshold=self.parameters['decision_threshold'],
            backspace_prob=self.parameters['lm_backspace_prob'],
            backspace_always_shown=self.parameters['backspace_always_shown'],
            filter_high=self.parameters['filter_high'],
            filter_low=self.parameters['filter_low'],
            filter_order=self.parameters['filter_order'],
            notch_filter_frequency=self.parameters['notch_filter_frequency'],
            stim_length=self.parameters['stim_length'])

    def await_start(self) -> bool:
        """Wait on the splash screen for the user to either exit or start."""
        self.logger.debug('Awaiting user start.')
        should_continue = get_user_input(
            self.rsvp,
            self.parameters['wait_screen_message'],
            self.parameters['wait_screen_message_color'],
            first_run=True)
        return should_continue

    def user_wants_to_continue(self) -> bool:
        """Check if user wants to continue or terminate.
        
        Returns
        -------
        - `True` to continue
        - `False` to finish the task.
        """
        should_continue = get_user_input(
            self.rsvp,
            self.parameters['wait_screen_message'],
            self.parameters['wait_screen_message_color'],
            first_run=False)
        if not should_continue:
            self.logger.debug('User wants to exit.')
        return should_continue

    def wait(self, seconds: float = None):
        """Pause for a time.

        Parameters
        ----------
        - seconds : duration of time to wait; if missing, defaults to the
        value of the parameter `'task_buffer_len'`
        """
        seconds = seconds or self.parameters['task_buffer_len']
        core.wait(seconds)

    def present_inquiry(self, inquiry_schedule: InquirySchedule
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
        self.rsvp.update_task_state(text=self.spelled_text,
                                    color_list=['white'])
        self.rsvp.draw_static()
        self.window.flip()

        # Setup the new Stimuli
        self.rsvp.stimuli_inquiry = inquiry_schedule.stims[0]
        if self.parameters['is_txt_stim']:
            self.rsvp.stimuli_colors = inquiry_schedule.colors[0]
        self.rsvp.stimuli_timing = inquiry_schedule.durations[0]

        self.wait()

        if self.parameters['show_preview_inquiry']:
            stim_times, proceed = self.rsvp.preview_inquiry()
            if proceed:
                stim_times.extend(self.rsvp.do_inquiry())
        else:
            stim_times = self.rsvp.do_inquiry()
            proceed = True

        return stim_times, proceed

    def show_feedback(self, selection: str, correct: bool = True):
        """Display the selection as feedback if the 'show_feedback'
        parameter is configured.

        Parameters
        ----------
        - selection : selected stimulus to display
        - correct : whether or not the correct stim was chosen
        """
        if self.parameters['show_feedback']:
            feedback = VisualFeedback(self.window, self.parameters,
                                      self.experiment_clock)
            feedback.administer(
                selection,
                message='Selected:',
                fill_color=self.parameters['feedback_message_color'])

    def stoppage_criteria_ok(self) -> bool:
        """Returns True if experiment is currently within params and the task
        should continue.
        """
        if self.copy_phrase == self.spelled_text:
            self.logger.debug('Spelling complete')
            return False

        if (self.inq_counter + 1) >= self.parameters['max_inq_len']:
            self.logger.debug('Max tries exceeded: to allow for more tries'
                              ' adjust the Maximum inquiry Length '
                              '(max_inq_len) parameter.')
            return False

        if self.session.total_time_spent >= (self.parameters['max_minutes'] *
                                             60):
            self.logger.debug('Max time exceeded. To allow for more time '
                              'adjust the max_minutes parameter.')
            return False
        return True

    def next_target(self) -> str:
        """Computes the next target letter based on the currently spelled_text.
        """
        if self.copy_phrase[0:len(self.spelled_text)] == self.spelled_text:
            # if correctly spelled so far, get the next letter.
            return self.copy_phrase[len(self.spelled_text)]
        return BACKSPACE_CHAR

    def execute(self) -> str:
        """Executes the task.

        Returns
        -------
        data save location (triggers.txt, session.json)
        """
        self.logger.debug('Starting Copy Phrase Task!')
        self.setup()

        with open(self.trigger_save_location, 'w') as trigger_file:
            run = self.await_start()

            while run and self.user_wants_to_continue():
                target_letter = self.next_target()
                stim_times, proceed = self.present_inquiry(
                    self.current_inquiry)

                self.write_trigger_data(stim_times, trigger_file)
                self.wait()

                evidence_types = self.add_evidence(stim_times, proceed)
                decision = self.evaluate_evidence()

                data = self.new_data_record(stim_times,
                                            target_letter,
                                            current_text=self.spelled_text,
                                            next_state=decision.spelled_text,
                                            evidence_types=evidence_types)
                self.update_session_data(data,
                                         save=True,
                                         decision_made=decision.decision_made)

                if decision.decision_made:
                    self.show_feedback(decision.selection,
                                       (decision.selection == target_letter))
                    self.spelled_text = decision.spelled_text
                    self.current_inquiry = self.next_inquiry()

                else:
                    self.current_inquiry = decision.new_inq_schedule

                run = self.stoppage_criteria_ok()
                self.inq_counter += 1

            self.exit_display()
            self.write_offset_trigger(trigger_file)

        # Wait some time before exiting so there is trailing eeg data saved
        self.wait(seconds=self.parameters['eeg_buffer_len'])

        return self.file_save

    def evaluate_evidence(self) -> Decision:
        """Uses the `copy_phrase_task` parameter to evaluate the provided
        evidence and attempt a decision. 
        
        Modifies
        --------
        - self.copy_phrase_task
        """
        if self.fake:
            _, spelled, _ = fake_copy_phrase_decision(self.copy_phrase,
                                                      self.next_target(),
                                                      self.spelled_text)
            # Reset the stoppage criteria by forcing the commit to a decision.
            self.copy_phrase_task.decision_maker.do_series()

            # In fake mode, all inquiries result in a selection.
            return Decision(decision_made=True,
                            selection=spelled[-1],
                            spelled_text=spelled,
                            new_inq_schedule=None)

        decision_made, new_sti = self.copy_phrase_task.decide()
        spelled_text = self.copy_phrase_task.decision_maker.displayed_state
        selection = self.copy_phrase_task.decision_maker.last_selection

        return Decision(decision_made, selection, spelled_text, new_sti)

    def add_evidence(self, stim_times: List[List],
                     proceed: bool = True) -> List[str]:
        """Add all evidence used to make a decision.

        Parameters
        ----------
        - stim_times : list of stimuli returned from the display
        - proceed : whether or not to proceed with the inquiry
        
        Returns
        -------
        list of evidence types added

        Modifies
        --------
        - self.copy_phrase_task
        """
        evidences = [
            self.compute_button_press_evidence(proceed),
            self.compute_eeg_evidence(stim_times, proceed)
        ]
        evidence_types = []
        for evidence in evidences:
            if evidence:
                name, probs = evidence
                evidence_types.append(name)
                self.copy_phrase_task.add_evidence(name, probs)
        if self.session.latest_series_is_empty():
            evidence_types.append(EvidenceName.LM)
        return evidence_types

    def compute_button_press_evidence(self, proceed: bool
                                      ) -> Tuple[str, List[float]]:
        """If 'show_preview_inquiry' feature is enabled, compute the button
        press evidence and add it to the copy phrase task.

        Parameters
        ----------
        - proceed : whether to proceed with the inquiry after the preview

        Returns
        -------
        tuple of (evidence name, evidence)
        """
        if not self.parameters['show_preview_inquiry']:
            return None
        probs = compute_probs_after_preview(self.current_inquiry.stims[0],
                                            self.alp,
                                            self.button_press_error_prob,
                                            proceed)
        return (EvidenceName.BTN, probs)

    def compute_eeg_evidence(self,
                             stim_times: List[List],
                             proceed: bool = True) -> Tuple[str, List[float]]:
        """Evaluate the EEG evidence and add it to the copy_phrase_task, but
        don't yet attempt a decision.

        Parameters
        ----------
        - stim_times : list of stimuli returned from the display
        - proceed : whether or not to evaluate the evidence, if `False` returns
        empty values.

        Returns
        -------
        tuple of (evidence name, evidence)
        """
        if not proceed or self.fake:
            return None

        raw_data, triggers, target_info = process_data_for_decision(
            self.stims_for_decision(stim_times), self.daq, self.window,
            self.parameters, self.rsvp.first_stim_time,
            self.parameters['static_trigger_offset'])

        probs = self.copy_phrase_task.evaluate_eeg_evidence(
            raw_data, triggers, target_info, self.parameters['trial_length'])
        return (EvidenceName.ERP, probs)

    def stims_for_decision(self, stim_times: List[List]) -> List[List]:
        """The stim_timings from the display may include non-letter stimuli
        such as calibration and inquiry_preview timings. This method extracts
        only the letter data used to process the data for a decision.
        
        Parameters
        ----------
        - stim_times : list of [stim, clock_time] pairs returned from display.

        Returns
        -------
        stim times where the stim is in the current alphabet; filters out
        'calibration', 'inquiry_preview', etc.
        """
        return [
            timing for timing in stim_times if timing[0] in (self.alp + ['+'])
        ]

    def new_data_record(self,
                        stim_times: List[List],
                        target_letter: str,
                        current_text: str,
                        next_state: str,
                        evidence_types: List[str] = []) -> Inquiry:
        """Construct a new inquiry data record.
        
        Parameters
        ----------
        - stim_times : list of [stim, clock_time] pairs returned from display.
        - target_letter : stim the user is currently attempting to spell.
        - current_text : spelled text before the inquiry
        - next_state : spelled text after consulting the decision maker
        - evidence_types : evidence provided to the decision-maker during the
        current inquiry.

        Returns
        -------
        Inquiry data for the current schedule; returned value only contains
        evidence for the provided evidence_types, leaving the other types empty
        """
        triggers = construct_triggers(self.stims_for_decision(stim_times))
        data = Inquiry(stimuli=self.current_inquiry.stims,
                       timing=self.current_inquiry.durations,
                       triggers=triggers,
                       target_info=target_info(triggers, target_letter),
                       target_letter=target_letter,
                       current_text=current_text,
                       target_text=self.copy_phrase,
                       next_display_state=next_state)

        if not self.fake:
            latest_evidence = self.copy_phrase_task.conjugator.latest_evidence
            data.evidences = {
                name: evidence if name in evidence_types else []
                for name, evidence in latest_evidence.items()
            }
            data.likelihood = list(self.copy_phrase_task.conjugator.likelihood)
        return data

    def exit_display(self):
        """Close the UI and cleanup."""
        # Update task state and reset the static
        self.rsvp.update_task_state(text=self.spelled_text,
                                    color_list=['white'])
        # Say Goodbye!
        self.rsvp.text = trial_complete_message(self.window, self.parameters)
        self.rsvp.draw_static()
        self.window.flip()

        # Give the system time to process
        self.wait()

    def update_session_data(self,
                            data: Inquiry,
                            save: bool = True,
                            decision_made: bool = False) -> None:
        """Update the current session with the latest inquiry data

        Parameters
        ----------
        - data : inquiry to append to the session
        - save : if True, persists the session to disk.

        Modifies
        --------
        - self.session
        """
        self.session.add_sequence(data)
        self.session.total_time_spent = self.experiment_clock.getTime()
        if save:
            self.write_session_data()
        if decision_made:
            self.session.add_series()

    def write_session_data(self) -> None:
        """Save session data to disk."""
        if self.session:
            _save_session_related_data(self.session_save_location,
                                       self.session.as_dict())

    def write_offset_trigger(self, trigger_file: TextIO):
        """Append the offset to the end of the triggers file.

        Parameters
        -----------
        - trigger_file : open file in which to write
        """
        if self.daq.is_calibrated:
            _write_triggers_from_inquiry_copy_phrase(
                ['offset', self.daq.offset],
                trigger_file,
                self.copy_phrase,
                self.spelled_text,
                offset=True)

    def write_trigger_data(self, stim_times: List[Tuple[str, float]],
                           trigger_file: TextIO) -> None:
        """Save trigger data to disk.

        Parameters
        ----------
        - stim_times : list of (stim, clock_time) tuples
        - trigger_file : data will be appended to this file
        """
        # TODO: fix the helper to correctly handle the targetness of inquiry_preview stims.
        _write_triggers_from_inquiry_copy_phrase(stim_times, trigger_file,
                                                 self.copy_phrase,
                                                 self.spelled_text)

    def name(self) -> str:
        return self.TASK_NAME


def _init_copy_phrase_display(parameters, win, daq, static_clock,
                              experiment_clock):
    preview_inquiry = PreviewInquiryProperties(
        preview_inquiry_length=parameters['preview_inquiry_length'],
        preview_inquiry_key_input=parameters['preview_inquiry_key_input'],
        preview_inquiry_progress_method=parameters[
            'preview_inquiry_progress_method'],
        preview_inquiry_isi=parameters['preview_inquiry_isi'])
    info = InformationProperties(
        info_color=parameters['info_color'],
        info_pos=(parameters['text_pos_x'], parameters['text_pos_y']),
        info_height=parameters['info_height'],
        info_font=parameters['info_font'],
        info_text=parameters['info_text'],
    )
    stimuli = StimuliProperties(stim_font=parameters['stim_font'],
                                stim_pos=(parameters['stim_pos_x'],
                                          parameters['stim_pos_y']),
                                stim_height=parameters['stim_height'],
                                stim_inquiry=['a'] * 10,
                                stim_colors=[parameters['stim_color']] * 10,
                                stim_timing=[3] * 10,
                                is_txt_stim=parameters['is_txt_stim'])
    task_display = TaskDisplayProperties(task_color=[parameters['task_color']],
                                         task_pos=(-.8, .9),
                                         task_font=parameters['task_font'],
                                         task_height=parameters['task_height'],
                                         task_text='****')
    return CopyPhraseDisplay(win,
                             static_clock,
                             experiment_clock,
                             stimuli,
                             task_display,
                             info,
                             marker_writer=daq.marker_writer,
                             static_task_text=parameters['task_text'],
                             static_task_color=parameters['task_color'],
                             trigger_type=parameters['trigger_type'],
                             space_char=parameters['stim_space_char'],
                             preview_inquiry=preview_inquiry)


def _init_copy_phrase_wrapper(min_num_inq, max_num_inq, signal_model, fs, k,
                              alp, evidence_names, task_list, lmodel,
                              is_txt_stim, device_name, device_channels,
                              stimuli_timing, decision_threshold,
                              backspace_prob, backspace_always_shown,
                              filter_high, filter_low, filter_order,
                              notch_filter_frequency, stim_length):
    return CopyPhraseWrapper(min_num_inq,
                             max_num_inq,
                             signal_model=signal_model,
                             fs=fs,
                             k=k,
                             alp=alp,
                             evidence_names=evidence_names,
                             task_list=task_list,
                             lmodel=lmodel,
                             is_txt_stim=is_txt_stim,
                             device_name=device_name,
                             device_channels=device_channels,
                             stimuli_timing=stimuli_timing,
                             decision_threshold=decision_threshold,
                             backspace_prob=backspace_prob,
                             backspace_always_shown=backspace_always_shown,
                             filter_high=filter_high,
                             filter_low=filter_low,
                             filter_order=filter_order,
                             notch_filter_frequency=notch_filter_frequency,
                             stim_length=stim_length)
