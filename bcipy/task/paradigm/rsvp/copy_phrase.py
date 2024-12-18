# mypy: disable-error-code="arg-type"
import logging
from typing import Any, List, NamedTuple, Optional, Tuple

from psychopy import core, visual
from psychopy.visual import Window

from bcipy.acquisition import ClientManager
from bcipy.config import (DEFAULT_EVIDENCE_PRECISION, SESSION_DATA_FILENAME,
                          SESSION_LOG_FILENAME, SESSION_SUMMARY_FILENAME,
                          TRIGGER_FILENAME, WAIT_SCREEN_MESSAGE)
from bcipy.display import (InformationProperties, StimuliProperties,
                           init_display_window)
from bcipy.display.components.task_bar import CopyPhraseTaskBar
from bcipy.display.main import PreviewParams
from bcipy.display.paradigm.rsvp.mode.copy_phrase import CopyPhraseDisplay
from bcipy.exceptions import TaskConfigurationException
from bcipy.feedback.visual.visual_feedback import VisualFeedback
from bcipy.helpers.acquisition import (LslDataServer, active_content_types,
                                       init_acquisition)
from bcipy.helpers.clock import Clock
from bcipy.helpers.copy_phrase_wrapper import CopyPhraseWrapper
from bcipy.helpers.language_model import init_language_model
from bcipy.helpers.list import destutter
from bcipy.helpers.load import choose_signal_models
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.save import _save_session_related_data
from bcipy.helpers.session import session_excel
from bcipy.helpers.stimuli import InquirySchedule, StimuliOrder
from bcipy.helpers.symbols import BACKSPACE_CHAR, alphabet
from bcipy.helpers.task import (consecutive_incorrect, construct_triggers,
                                fake_copy_phrase_decision,
                                get_device_data_for_decision, get_user_input,
                                relative_triggers, target_info,
                                trial_complete_message)
from bcipy.helpers.triggers import (FlushFrequency, Trigger, TriggerHandler,
                                    TriggerType, convert_timing_triggers,
                                    offset_label)
from bcipy.language.main import LanguageModel
from bcipy.signal.model import SignalModel
from bcipy.signal.model.inquiry_preview import compute_probs_after_preview
from bcipy.task import Task, TaskData, TaskMode
from bcipy.task.control.evidence import (EvidenceEvaluator,
                                         init_evidence_evaluator)
from bcipy.task.data import EvidenceType, Inquiry, Session
from bcipy.task.exceptions import DuplicateModelEvidence

logger = logging.getLogger(SESSION_LOG_FILENAME)


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
    new_inq_schedule: Optional[InquirySchedule]


class RSVPCopyPhraseTask(Task):
    """RSVP Copy Phrase Task.

    Initializes and runs all needed code for executing a copy phrase task. A
        phrase is set in parameters and necessary objects (daq, display) are
        passed to this function. Certain Wrappers and Task Specific objects are
        executed here.

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

    name = "RSVP Copy Phrase"
    paradigm = "RSVP"
    mode = TaskMode.COPYPHRASE
    initalized = False

    PARAMETERS_USED = [
        'backspace_always_shown',
        'decision_threshold',
        'down_sampling_rate',
        'feedback_duration',
        'filter_high',
        'filter_low',
        'filter_order',
        'fixation_color',
        'font',
        'info_color',
        'info_color',
        'info_height',
        'info_height',
        'info_pos_x', 'info_pos_y',
        'info_text',
        'info_text',
        'is_txt_stim',
        'lm_backspace_prob',
        'max_incorrect',
        'max_inq_len',
        'max_inq_per_series',
        'max_minutes',
        'max_selections',
        'min_inq_len',
        'notch_filter_frequency',
        'prestim_length',
        'preview_box_text_size',
        'preview_inquiry_error_prob',
        'preview_inquiry_isi',
        'preview_inquiry_key_input',
        'preview_inquiry_length',
        'preview_inquiry_progress_method',
        'rsvp_stim_height',
        'rsvp_stim_pos_x', 'rsvp_stim_pos_y',
        'rsvp_task_height',
        'rsvp_task_padding',
        'show_feedback',
        'show_preview_inquiry',
        'spelled_letters_count',
        'stim_color',
        'stim_jitter',
        'stim_length',
        'stim_number',
        'stim_order',
        'stim_space_char',
        'target_color',
        'task_buffer_length',
        'task_color',
        'task_text',
        'time_fixation',
        'time_flash',
        'time_prompt',
        'trial_window',
        'trigger_type',
    ]

    def __init__(
        self,
        parameters: Parameters,
        file_save: str,
        fake: bool = False,
        **kwargs: Any
    ) -> None:
        super(RSVPCopyPhraseTask, self).__init__()
        self.fake = fake
        self.parameters = parameters
        self.language_model = self.get_language_model()
        self.signal_models = self.get_signal_models()
        daq, servers, win = self.setup(parameters, file_save, fake)
        self.servers = servers
        self.window = win
        self.daq = daq

        self.validate_parameters()

        self.static_clock = None
        self.experiment_clock = Clock()
        self.start_time = self.experiment_clock.getTime()

        self.alp = alphabet(self.parameters)

        self.button_press_error_prob = parameters['preview_inquiry_error_prob']

        self.signal_model = self.signal_models[0] if self.signal_models else None
        self.evidence_evaluators = self.init_evidence_evaluators(self.signal_models)
        self.evidence_types = self.init_evidence_types(self.signal_models, self.evidence_evaluators)

        self.file_save = file_save
        self.save_session_every_inquiry = True

        self.trigger_handler = self.default_trigger_handler()
        self.session_save_location = f"{self.file_save}/{SESSION_DATA_FILENAME}"
        self.copy_phrase = parameters["task_text"]

        self.evidence_precision = DEFAULT_EVIDENCE_PRECISION

        self.feedback = self.init_feedback()

        self.set()

        self.rsvp = self.init_display()

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
                logger.exception(f'Cannot load signal model. Exiting. {error}')
                raise error
            return signal_models
        return []

    def cleanup(self):
        self.exit_display()
        self.write_offset_trigger()
        self.save_session_data()
        # Wait some time before exiting so there is trailing eeg data saved
        self.wait()

        if self.initalized:

            try:
                # Stop Acquisition
                self.daq.stop_acquisition()
                self.daq.cleanup()

                # Stop Servers
                if self.servers:
                    for server in self.servers:
                        server.stop()

                # Close the display window
                # NOTE: There is currently a bug in psychopy when attempting to shutdown
                # windows when using a USB-C monitor. Putting the display close last in
                # the inquiry allows acquisition to properly shutdown.
                self.window.close()
                self.initalized = False

            except Exception as e:
                logger.exception(str(e))

    def save_session_data(self) -> None:
        self.session.task_summary = TaskSummary(
            self.session,
            self.parameters["show_preview_inquiry"],
            self.parameters["preview_inquiry_progress_method"],
            self.trigger_handler.file_path,
        ).as_dict()
        self.write_session_data()

        # Evidence is not recorded in the session when using fake decisions.
        if self.parameters["summarize_session"] and self.session.has_evidence():
            session_excel(
                session=self.session,
                excel_file=f"{self.file_save}/{SESSION_SUMMARY_FILENAME}",
            )

    def init_evidence_evaluators(
        self, signal_models: List[SignalModel]
    ) -> List[EvidenceEvaluator]:
        """Initializes the evidence evaluators from the provided signal models.

        Returns a list of evaluators for active devices. Raises an exception if
        more than one evaluator provides the same type of evidence.
        """
        evidence_types = []
        evaluators = []
        for model in signal_models:
            evaluator = init_evidence_evaluator(self.alp, model)
            content_type = evaluator.consumes
            evidence_type = evaluator.produces
            if content_type in self.daq.active_device_content_types:
                evaluators.append(evaluator)
                if evidence_type in evidence_types:
                    raise DuplicateModelEvidence(
                        f"More than one model produces {evidence_type} evidence"
                    )
                evidence_types.append(evidence_type)
            else:
                logger.info(
                    f"SignalModel not used: there is no active device of type: {content_type}"
                )
        return evaluators

    def init_evidence_types(
            self, signal_models: List[SignalModel],
            evidence_evaluators: List[EvidenceEvaluator]
    ) -> List[EvidenceType]:
        evidence_types = [EvidenceType.LM]
        evidence_types.extend(
            [evaluator.produces for evaluator in evidence_evaluators])
        if self.parameters['show_preview_inquiry']:
            evidence_types.append(EvidenceType.BTN)
        return evidence_types

    def default_trigger_handler(self) -> TriggerHandler:
        """Default trigger handler"""
        return TriggerHandler(self.file_save, TRIGGER_FILENAME,
                              FlushFrequency.EVERY)

    def set(self) -> None:
        """Initialize/reset parameters used in the execute run loop."""

        self.spelled_text = str(self.copy_phrase[0: self.starting_spelled_letters()])
        self.last_selection = ""
        self.inq_counter = 0
        self.session = Session(
            save_location=self.file_save,
            task="Copy Phrase",
            mode=str(self.mode),
            symbol_set=self.alp,
            decision_threshold=self.parameters["decision_threshold"],
        )
        self.write_session_data()

        self.init_copy_phrase_task()
        self.current_inquiry = self.next_inquiry()

    def init_display(self) -> CopyPhraseDisplay:
        """Initialize the display"""
        return _init_copy_phrase_display(
            self.parameters,
            self.window,
            self.static_clock,
            self.experiment_clock,
            self.spelled_text,
        )

    def init_feedback(self) -> Optional[VisualFeedback]:
        """Initialize visual feedback"""
        return VisualFeedback(
            self.window, {
                'feedback_font': self.parameters['font'],
                'feedback_color': self.parameters['info_color'],
                'feedback_pos_x': self.parameters['info_pos_x'],
                'feedback_pos_y': self.parameters['info_pos_y'],
                'feedback_stim_height': self.parameters['info_height'],
                'feedback_duration': self.parameters['feedback_duration']
            }, self.experiment_clock)

    def validate_parameters(self) -> None:
        """Validate.

        Confirm Task is configured with correct parameters and within operating limits.
        """

        # ensure all required parameters are provided
        for param in RSVPCopyPhraseTask.PARAMETERS_USED:
            if param not in self.parameters:
                raise TaskConfigurationException(f"parameter '{param}' is required")

        # ensure data / query parameters are set correctly
        buffer_len = self.parameters["task_buffer_length"]
        prestim = self.parameters["prestim_length"]
        poststim = (
            self.parameters["trial_window"][1] - self.parameters["trial_window"][0]
        )
        if buffer_len < prestim:
            raise TaskConfigurationException(
                f"task_buffer_length=[{buffer_len}] must be greater than prestim_length=[{prestim}]"
            )

        if buffer_len < poststim:
            raise TaskConfigurationException(
                f"task_buffer_length=[{buffer_len}] must be greater than trial_length=[{poststim}]"
            )

    def starting_spelled_letters(self) -> int:
        """Number of letters already spelled at the start of the task."""
        spelled_letters_count = self.parameters["spelled_letters_count"]
        if spelled_letters_count > len(self.copy_phrase):
            logger.info("Already spelled letters exceeds phrase length.")
            spelled_letters_count = 0
        return spelled_letters_count

    def next_inquiry(self) -> Optional[InquirySchedule]:
        """Generate an InquirySchedule for spelling the next letter."""
        if self.copy_phrase_task:
            _, inquiry_schedule = self.copy_phrase_task.initialize_series()
            return inquiry_schedule
        return None

    def init_copy_phrase_task(self) -> None:
        """Initialize the CopyPhraseWrapper.

        Returns:
        --------
        initialized CopyPhraseWrapper
        """

        self.copy_phrase_task = CopyPhraseWrapper(
            self.parameters["min_inq_len"],
            self.parameters["max_inq_per_series"],
            lmodel=self.language_model,
            alp=self.alp,
            evidence_names=self.evidence_types,
            task_list=[(str(self.copy_phrase), self.spelled_text)],
            is_txt_stim=self.parameters["is_txt_stim"],
            stim_timing=[
                self.parameters["time_fixation"],
                self.parameters["time_flash"],
            ],
            decision_threshold=self.parameters["decision_threshold"],
            backspace_prob=self.parameters["lm_backspace_prob"],
            backspace_always_shown=self.parameters["backspace_always_shown"],
            stim_length=self.parameters["stim_length"],
            stim_jitter=self.parameters["stim_jitter"],
            stim_order=StimuliOrder(self.parameters["stim_order"]),
        )

    def user_wants_to_continue(self) -> bool:
        """Check if user wants to continue or terminate.

        Returns
        -------
        - `True` to continue
        - `False` to finish the task.
        """
        should_continue = get_user_input(
            self.rsvp,
            WAIT_SCREEN_MESSAGE,
            self.parameters["stim_color"],
            first_run=self.first_run,
        )
        if not should_continue:
            logger.info("User wants to exit.")
        return should_continue

    def wait(self, seconds: Optional[float] = None) -> None:
        """Pause for a time.

        Parameters
        ----------
        - seconds : duration of time to wait; if missing, defaults to the
        value of the parameter `'task_buffer_length'`
        """
        seconds = seconds or self.parameters["task_buffer_length"]
        core.wait(seconds)

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
        self.rsvp.update_task_bar(self.spelled_text)
        self.rsvp.draw_static()
        self.window.flip()

        self.wait()

        # Setup the new stimuli
        self.rsvp.schedule_to(
            stimuli=inquiry_schedule.stimuli[0],
            timing=inquiry_schedule.durations[0],
            colors=(
                inquiry_schedule.colors[0] if self.parameters["is_txt_stim"] else None
            ),
        )

        stim_times = self.rsvp.do_inquiry()
        proceed = not self.rsvp.preview_enabled or self.rsvp.preview_accepted
        return stim_times, proceed

    def show_feedback(self, selection: str, correct: bool = True) -> None:
        """Display the selection as feedback if the 'show_feedback'
        parameter is configured.

        Parameters
        ----------
        - selection : selected stimulus to display
        - correct : whether or not the correct stim was chosen
        """
        if self.parameters['show_feedback'] and self.feedback:
            self.feedback.administer(f'Selected: {selection}')

    def check_stop_criteria(self) -> bool:
        """Returns True if experiment is currently within params and the task
        should continue.
        """
        if self.copy_phrase == self.spelled_text:
            logger.info("Spelling complete")
            return False

        if (self.inq_counter + 1) >= self.parameters["max_inq_len"]:
            logger.info(
                "Max tries exceeded: to allow for more tries"
                " adjust the Maximum inquiry Length "
                "(max_inq_len) parameter."
            )
            return False

        if self.session.total_time_spent >= (self.parameters["max_minutes"] * 60):
            logger.info(
                "Max time exceeded. To allow for more time "
                "adjust the max_minutes parameter."
            )
            return False

        if self.session.total_number_decisions >= self.parameters["max_selections"]:
            logger.info(
                "Max number of selections reached "
                "(configured with the max_selections parameter)"
            )
            return False

        if consecutive_incorrect(
                target_text=self.copy_phrase,
                spelled_text=self.spelled_text) >= self.parameters.get(
                    'max_incorrect'):
            logger.info(
                'Max number of consecutive incorrect selections reached '
                '(configured with the max_incorrect parameter)')
            return False

        return True

    def next_target(self) -> str:
        """Computes the next target letter based on the currently spelled_text."""
        if self.copy_phrase[0: len(self.spelled_text)] == self.spelled_text:
            # if correctly spelled so far, get the next letter.
            return self.copy_phrase[len(self.spelled_text)]
        return BACKSPACE_CHAR

    def execute(self) -> TaskData:
        """Executes the task.

        Returns
        -------
        data save location (triggers.txt, session.json)
        """
        logger.info("Starting Copy Phrase Task!")
        run = True
        self.wait()  # buffer for data processing

        while run and self.user_wants_to_continue() and self.current_inquiry:
            target_stimuli = self.next_target()
            stim_times, proceed = self.present_inquiry(self.current_inquiry)

            self.write_trigger_data(stim_times, target_stimuli)
            self.wait()

            evidence_types = self.add_evidence(stim_times, proceed)
            decision = self.evaluate_evidence()

            data = self.new_data_record(stim_times,
                                        target_stimuli,
                                        current_text=self.spelled_text,
                                        decision=decision,
                                        evidence_types=evidence_types)
            self.update_session_data(data,
                                     save=self.save_session_every_inquiry,
                                     decision_made=decision.decision_made)

            if decision.decision_made:
                self.show_feedback(
                    decision.selection, (decision.selection == target_stimuli)
                )
                self.spelled_text = decision.spelled_text
                self.current_inquiry = self.next_inquiry()

            else:
                self.current_inquiry = decision.new_inq_schedule

            run = self.check_stop_criteria()
            self.inq_counter += 1

        self.cleanup()

        return TaskData(save_path=self.file_save, task_dict=self.session.as_dict())

    def evaluate_evidence(self) -> Decision:
        """Uses the `copy_phrase_task` parameter to evaluate the provided
        evidence and attempt a decision.

        Modifies
        --------
        - self.copy_phrase_task
        """
        if self.fake:
            _, spelled, _ = fake_copy_phrase_decision(
                self.copy_phrase, self.next_target(), self.spelled_text
            )
            # Reset the stoppage criteria by forcing the commit to a decision.
            self.copy_phrase_task.decision_maker.do_series()
            # In fake mode, only the LM is providing evidence, so the decision
            # made is the highest symbol predicted. Override this state
            self.copy_phrase_task.decision_maker.update(spelled)

            # In fake mode, all inquiries result in a selection.
            return Decision(
                decision_made=True,
                selection=spelled[-1],
                spelled_text=spelled,
                new_inq_schedule=None,
            )

        decision_made, new_sti = self.copy_phrase_task.decide()
        spelled_text = self.copy_phrase_task.decision_maker.displayed_state
        selection = ""
        if decision_made:
            selection = self.copy_phrase_task.decision_maker.last_selection

        return Decision(decision_made, selection, spelled_text, new_sti)

    def add_evidence(
        self, stim_times: List[List], proceed: bool = True
    ) -> List[EvidenceType]:
        """Add all evidence used to make a decision.

        Evaluates evidence from various sources (button press, devices,
        language model) and adds it to the CopyPhraseWrapper for evaluation.

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
        evidences = [self.compute_button_press_evidence(proceed)]
        # evidence from one or more devices
        evidences.extend(self.compute_device_evidence(stim_times, proceed))

        evidence_types = []
        for evidence in evidences:
            if evidence:
                evidence_type, probs = evidence
                evidence_types.append(evidence_type)
                self.copy_phrase_task.add_evidence(evidence_type, probs)
        if self.session.latest_series_is_empty():
            evidence_types.append(EvidenceType.LM)
        return evidence_types

    def compute_button_press_evidence(
            self, proceed: bool) -> Optional[Tuple[EvidenceType, List[float]]]:
        """If 'show_preview_inquiry' feature is enabled, compute the button
        press evidence and add it to the copy phrase task.

        Parameters
        ----------
            proceed : whether to proceed with the inquiry after the preview

        Returns
        -------
            tuple of (evidence type, evidence) or None if inquiry preview is
            not enabled.
        """
        if (not self.should_compute_button_press_evidence()):
            return None
        assert self.current_inquiry, "Current inquiry is required"
        probs = compute_probs_after_preview(
            self.current_inquiry.stimuli[0],
            self.alp,
            self.button_press_error_prob,
            proceed,
        )
        return (EvidenceType.BTN, probs)

    def should_compute_button_press_evidence(self) -> bool:
        """Determine if button press evidence should be computed"""
        return bool(self.parameters["show_preview_inquiry"] and self.parameters[
            'preview_inquiry_progress_method'] > 0 and self.current_inquiry)

    def compute_device_evidence(
            self,
            stim_times: List[List],
            proceed: bool = True) -> List[Tuple[EvidenceType, List[float]]]:
        """Get inquiry data from all devices and evaluate the evidence, but
        don't yet attempt a decision.

        Parameters
        ----------
        - stim_times : list of stimuli returned from the display
        - proceed : whether or not to evaluate the evidence, if `False` returns
            an empty array.

        Returns
        -------
            list of (evidence type, evidence) tuples
        """
        if not proceed or self.fake:
            return []

        # currently prestim_length is used as a buffer for filter application
        post_stim_buffer = int(self.parameters.get("task_buffer_length") / 2)
        prestim_buffer: float = self.parameters["prestim_length"]
        trial_window: Tuple[float, float] = self.parameters["trial_window"]
        window_length = trial_window[1] - trial_window[0]
        inquiry_timing = self.stims_for_decision(stim_times)

        # update the inquiry timing list (stim, time) based on the trial window first time value
        inquiry_timing = [
            (stim, time + trial_window[0]) for stim, time in inquiry_timing
        ]

        # Get all data at once so we don't redundantly query devices which are
        # used in more than one signal model.
        device_data = get_device_data_for_decision(
            inquiry_timing=inquiry_timing,
            daq=self.daq,
            prestim=prestim_buffer,
            poststim=post_stim_buffer + window_length,
        )

        triggers = relative_triggers(inquiry_timing, prestim_buffer)
        # we assume all are nontargets at this point
        labels = ["nontarget"] * len(triggers)
        letters, times, filtered_labels = self.copy_phrase_task.letter_info(
            triggers, labels
        )

        evidences = []
        for evidence_evaluator in self.evidence_evaluators:
            probs = evidence_evaluator.evaluate(
                raw_data=device_data[evidence_evaluator.consumes],
                symbols=letters,
                times=times,
                target_info=filtered_labels,
                window_length=window_length,
            )
            evidences.append((evidence_evaluator.produces, probs))

        return evidences

    def stims_for_decision(
        self, stim_times: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
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
        return [timing for timing in stim_times if timing[0] in (self.alp + ["+"])]

    def new_data_record(
        self,
        stim_times: List[Tuple[str, float]],
        target_stimuli: str,
        current_text: str,
        decision: Decision,
        evidence_types: Optional[List[EvidenceType]] = None,
    ) -> Inquiry:
        """Construct a new inquiry data record.

        Parameters
        ----------
        - stim_times : list of [stim, clock_time] pairs returned from display.
        - target_stimuli : stim the user is currently attempting to spell.
        - current_text : spelled text before the inquiry
        - decision : decision made by the decision maker
        - evidence_types : evidence provided to the decision-maker during the
        current inquiry.

        Returns
        -------
        Inquiry data for the current schedule; returned value only contains
        evidence for the provided evidence_types, leaving the other types empty
        """
        assert self.current_inquiry, "Current inquiry is required"
        evidence_types = evidence_types or []
        triggers = construct_triggers(self.stims_for_decision(stim_times))
        data = Inquiry(
            stimuli=self.current_inquiry.stimuli,
            timing=self.current_inquiry.durations,
            triggers=triggers,
            target_info=target_info(
                triggers, target_stimuli, self.parameters["is_txt_stim"]
            ),
            target_letter=target_stimuli,
            current_text=current_text,
            target_text=self.copy_phrase,
            selection=decision.selection,
            next_display_state=decision.spelled_text,
        )
        data.precision = self.evidence_precision

        if not self.fake:
            latest_evidence = self.copy_phrase_task.conjugator.latest_evidence
            data.evidences = {
                ev_type: evidence if ev_type in evidence_types else []
                for ev_type, evidence in latest_evidence.items()
            }
            data.likelihood = list(self.copy_phrase_task.conjugator.likelihood)
        return data

    def exit_display(self) -> None:
        """Close the UI and cleanup."""
        # Update task state and reset the static
        self.rsvp.update_task_bar(text=self.spelled_text)

        # Say Goodbye!
        self.rsvp.info_text = trial_complete_message(self.window, self.parameters)
        self.rsvp.draw_static()
        self.window.flip()

        # Give the system time to process
        self.wait()

    def update_session_data(
        self, data: Inquiry, save: bool = True, decision_made: bool = False
    ) -> None:
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
        self.session.total_time_spent = self.elapsed_seconds()
        if save:
            self.write_session_data()
        if decision_made:
            self.session.add_series()

    def elapsed_seconds(self) -> float:
        """Compute the number of seconds elapsed since the experiment start."""
        return self.experiment_clock.getTime() - self.start_time

    def write_session_data(self) -> None:
        """Save session data to disk."""
        if self.session:
            session_file = _save_session_related_data(
                self.session_save_location, self.session.as_dict()
            )
            session_file.close()

    def write_offset_trigger(self) -> None:
        """Append the offset to the end of the triggers file."""
        # To help support future refactoring or use of lsl timestamps only
        # we write only the sample offset here.
        triggers = []
        for content_type, client in self.daq.clients_by_type.items():
            label = offset_label(content_type.name, prefix="daq_sample_offset")
            time = client.offset(self.rsvp.first_stim_time)
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
                    client.offset(self.rsvp.first_stim_time) - self.rsvp.first_stim_time
                )
                offset_triggers.append(Trigger(label, TriggerType.OFFSET, time))
            self.trigger_handler.add_triggers(offset_triggers)

        triggers = convert_timing_triggers(
            stim_times, target_stimuli, self.trigger_type
        )
        self.trigger_handler.add_triggers(triggers)

    def trigger_type(self, symbol: str, target: str, index: int) -> TriggerType:
        """Trigger Type.

        Cast a given symbol to a TriggerType.
        """
        if symbol == "inquiry_preview":
            return TriggerType.PREVIEW
        if "key_press" in symbol:
            return TriggerType.EVENT
        if symbol == "+":
            return TriggerType.FIXATION
        if target == symbol:
            return TriggerType.TARGET
        return TriggerType.NONTARGET

    @property
    def first_run(self) -> bool:
        """First run.

        Determines whether it is the first inquiry presentation / run.
        """
        return self.inq_counter == 0


class TaskSummary:
    """Summary data for tracking performance metrics.

    Parameters
    ----------
        session - current session information
        show_preview - whether or not inquiry preview was displayed
        preview_mode - the switch mode for inquiry preview:
            0 = preview only;
            1 = press to confirm;
            2 = press to skip to another inquiry
    """

    def __init__(
        self,
        session: Session,
        show_preview: bool = False,
        preview_mode: int = 0,
        trigger_path: Optional[str] = None,
    ) -> None:
        assert preview_mode in range(3), "Preview mode out of range"
        self.session = session
        self.show_preview = show_preview
        self.preview_mode = preview_mode
        self.trigger_path = trigger_path
        logger = logging.getLogger(__name__)

    def as_dict(self) -> dict:
        """Computes the task summary data to append to the session."""

        selections = [inq for inq in self.session.all_inquiries if inq.selection]
        correct = [inq for inq in selections if inq.is_correct_decision]
        incorrect = [inq for inq in selections if not inq.is_correct_decision]

        # Note that SPACE is considered a symbol
        correct_symbols = [inq for inq in correct if inq.selection != BACKSPACE_CHAR]

        btn_presses = self.btn_press_count()
        sel_count = len(selections)
        switch_per_selection = (btn_presses / sel_count) if sel_count > 0 else 0
        accuracy = (len(correct) / sel_count) if sel_count > 0 else 0

        # Note that minutes includes startup time and any breaks.
        minutes = self.session.total_time_spent / 60
        return {
            "selections_correct": len(correct),
            "selections_incorrect": len(incorrect),
            "selections_correct_symbols": len(correct_symbols),
            "switch_total": btn_presses,
            "switch_per_selection": switch_per_selection,
            "switch_response_time": self.switch_response_time(),
            "typing_accuracy": accuracy,
            "correct_rate": len(correct) / minutes if minutes else 0,
            "copy_rate": len(correct_symbols) / minutes if minutes else 0,
        }

    def btn_press_count(self) -> int:
        """Compute the number of times the switch was activated. Returns 0 if
        inquiry preview mode was off or mode was preview-only."""

        if not self.show_preview or self.preview_mode == 0:
            return 0

        inquiries = self.session.all_inquiries
        activations = []
        if self.preview_mode == 1:
            # press to confirm
            activations = [inq for inq in inquiries if inq.eeg_evidence]
        elif self.preview_mode == 2:
            # press to skip
            activations = [inq for inq in inquiries if not inq.eeg_evidence]
        return len(activations)

    def switch_response_time(self) -> Optional[float]:
        """Computes the average switch response in seconds."""

        # Remove consecutive items with the same type; we are only interested
        # in PREVIEW followed by a EVENT.
        triggers = destutter(self.switch_triggers(), key=lambda trg: trg.type)
        pairs = list(zip(triggers[::2], triggers[1::2]))

        # Confirm that the data is structured as expected.
        for preview, keypress in pairs:
            if (preview.type != TriggerType.PREVIEW) or (
                keypress.type != TriggerType.EVENT
            ):
                logger.info("Could not compute switch_response_time")
                return None

        response_times = [keypress.time - preview.time for preview, keypress in pairs]
        count = len(response_times)
        return sum(response_times) / count if count > 0 else None

    def switch_triggers(self) -> List[Trigger]:
        """Returns a list of switch-related triggers"""
        if not self.trigger_path:
            return []
        triggers, _offset = TriggerHandler.read_text_file(self.trigger_path)
        return [
            trg
            for trg in triggers
            if trg.type in [TriggerType.PREVIEW, TriggerType.EVENT]
        ]


def _init_copy_phrase_display(
    parameters: Parameters,
    win: visual.Window,
    static_clock: core.StaticPeriod,
    experiment_clock: Clock,
    starting_spelled_text,
) -> CopyPhraseDisplay:
    info = InformationProperties(
        info_color=[parameters["info_color"]],
        info_pos=[(parameters["info_pos_x"], parameters["info_pos_y"])],
        info_height=[parameters["info_height"]],
        info_font=[parameters["font"]],
        info_text=[parameters["info_text"]],
    )
    stimuli = StimuliProperties(
        stim_font=parameters["font"],
        stim_pos=(parameters["rsvp_stim_pos_x"], parameters["rsvp_stim_pos_y"]),
        stim_height=parameters["rsvp_stim_height"],
        stim_inquiry=["A"] * parameters["stim_length"],
        stim_colors=[parameters["stim_color"]] * parameters["stim_length"],
        stim_timing=[10] * parameters["stim_length"],
        is_txt_stim=parameters["is_txt_stim"],
    )

    task_bar = CopyPhraseTaskBar(
        win,
        task_text=parameters["task_text"],
        spelled_text=starting_spelled_text,
        colors=[parameters["task_color"]],
        font=parameters["font"],
        height=parameters["rsvp_task_height"],
        padding=parameters["rsvp_task_padding"],
    )

    return CopyPhraseDisplay(
        win,
        static_clock,
        experiment_clock,
        stimuli,
        task_bar,
        info,
        starting_spelled_text,
        trigger_type=parameters['trigger_type'],
        space_char=parameters['stim_space_char'],
        preview_config=parameters.instantiate(PreviewParams))
