"""RSVP display module.

This module provides the base RSVP (Rapid Serial Visual Presentation) display implementation
which handles the visual presentation of stimuli during RSVP tasks. It provides core functionality
for stimulus presentation, timing control, and inquiry management.
"""

import logging
import os.path as path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from psychopy import core, visual

from bcipy.core.stimuli import resize_image
from bcipy.core.symbols import SPACE_CHAR
from bcipy.core.triggers import TriggerCallback, _calibration_trigger
from bcipy.display import (BCIPY_LOGO_PATH, Display, InformationProperties,
                           StimuliProperties)
from bcipy.display.components.task_bar import TaskBar
from bcipy.display.main import PreviewParams, init_preview_button_handler
from bcipy.helpers.clock import Clock
from bcipy.helpers.utils import get_screen_info


class RSVPDisplay(Display):
    """RSVP Display Object for inquiry Presentation.

    Animates display objects common to any RSVP task. Handles stimulus presentation,
    timing control, and inquiry management for RSVP-based BCI paradigms.

    Attributes:
        window (visual.Window): PsychoPy window for display.
        window_size (Tuple[float, float]): Size of the display window.
        refresh_rate (float): Display refresh rate.
        stimuli_inquiry (List[str]): List of stimuli to present.
        stimuli_colors (List[str]): List of colors for each stimulus.
        stimuli_timing (List[float]): List of presentation durations.
        stimuli_font (str): Font to use for text stimuli.
        textbox_font (str): Font to use for text boxes.
        stimuli_height (float): Height of stimuli.
        stimuli_pos (Tuple[float, float]): Position of stimuli.
        is_txt_stim (bool): Whether stimuli are text-based.
        stim_length (int): Length of stimulus list.
        full_screen (bool): Whether display is fullscreen.
        preview_params (Optional[PreviewParams]): Preview configuration.
        preview_button_handler (Optional[Any]): Handler for preview buttons.
        preview_accepted (bool): Whether preview was accepted.
        staticPeriod (core.StaticPeriod): Clock for static timing.
        first_run (bool): Whether this is the first run.
        first_stim_time (Optional[float]): Time of first stimulus.
        trigger_type (str): Type of trigger to use.
        trigger_callback (TriggerCallback): Callback for triggers.
        experiment_clock (Clock): Clock for experiment timing.
        first_stim_callback (Callable): Callback for first stimulus.
        size_list_sti (List[float]): List of stimulus sizes.
        space_char (str): Character to use for spaces.
        task_bar (TaskBar): Task bar component.
        info (InformationProperties): Information display properties.
        info_text (List[visual.TextStim]): Text stimuli for information.
        sti (List[visual.TextStim]): Initial stimuli objects.
    """

    def __init__(
            self,
            window: visual.Window,
            static_clock: core.StaticPeriod,
            experiment_clock: Clock,
            stimuli: StimuliProperties,
            task_bar: TaskBar,
            info: InformationProperties,
            preview_config: Optional[PreviewParams] = None,
            trigger_type: str = 'image',
            space_char: str = SPACE_CHAR,
            full_screen: bool = False) -> None:
        """Initialize RSVP display parameters and objects.

        Args:
            window (visual.Window): PsychoPy Window for display.
            static_clock (core.StaticPeriod): Used to schedule static periods of display time.
            experiment_clock (Clock): Clock used to timestamp display onsets.
            stimuli (StimuliProperties): Attributes used for inquiries.
            task_bar (TaskBar): Used for task tracking. Ex. 1/100.
            info (InformationProperties): Attributes to display informational stimuli.
            preview_config (Optional[PreviewParams]): Parameters for preview functionality.
                If None, preview is not displayed.
            trigger_type (str, optional): Defines the calibration trigger type. Defaults to 'image'.
            space_char (str, optional): Character to use for spaces. Defaults to SPACE_CHAR.
            full_screen (bool, optional): Whether window is fullscreen. Defaults to False.
        """
        self.window = window
        self.window_size = self.window.size  # [w, h]
        self.refresh_rate = window.getActualFrameRate()

        self.logger = logging.getLogger(__name__)

        # Stimuli parameters
        self.stimuli_inquiry = stimuli.stim_inquiry
        self.stimuli_colors = stimuli.stim_colors
        self.stimuli_timing = stimuli.stim_timing
        self.stimuli_font = stimuli.stim_font
        self.textbox_font = 'Consolas'  # Avoid TextBox2 font bug
        self.stimuli_height = stimuli.stim_height
        self.stimuli_pos = stimuli.stim_pos
        self.is_txt_stim = stimuli.is_txt_stim
        self.stim_length = stimuli.stim_length

        self.full_screen = full_screen

        self.preview_params = preview_config
        self.preview_button_handler = init_preview_button_handler(
            preview_config, experiment_clock) if self.preview_enabled else None
        self.preview_accepted = True

        self.staticPeriod = static_clock

        # Trigger handling
        self.first_run = True
        self.first_stim_time = None
        self.trigger_type = trigger_type
        self.trigger_callback = TriggerCallback()
        self.experiment_clock = experiment_clock

        # Callback used on presentation of first stimulus.
        self.first_stim_callback: Callable = lambda _sti: None
        self.size_list_sti: List[float] = []
        self.space_char = space_char

        self.task_bar = task_bar

        # Create multiple text objects based on input
        self.info = info
        self.info_text = info.build_info_text(window)

        # Create initial stimuli object for updating
        self.sti = stimuli.build_init_stimuli(window)

    @property
    def preview_enabled(self) -> bool:
        """Check if inquiry preview should be enabled.

        Returns:
            bool: True if preview is enabled and configured, False otherwise.
        """
        return bool(self.preview_params and self.preview_params.show_preview_inquiry)

    def draw_static(self) -> None:
        """Draw static elements in a stimulus."""
        if self.task_bar:
            self.task_bar.draw()
        for info in self.info_text:
            info.draw()

    def schedule_to(self,
                    stimuli: Optional[List[str]] = None,
                    timing: Optional[List[float]] = None,
                    colors: Optional[List[str]] = None) -> None:
        """Schedule stimuli elements (works as a buffer).

        Args:
            stimuli (Optional[List[str]]): List of stimuli text/name.
            timing (Optional[List[float]]): List of timings of stimuli.
            colors (Optional[List[str]]): List of colors.
        """
        self.stimuli_inquiry = stimuli or []
        self.stimuli_timing = timing or []
        self.stimuli_colors = colors or []

    @property
    def preview_index(self) -> int:
        """Get index within an inquiry at which the preview should be displayed.

        For calibration, we should display it after the target prompt (index = 1).
        For copy phrase there is no target prompt so it should display before the
        rest of the inquiry.

        Returns:
            int: The index at which to display the preview.
        """
        return 1

    def do_inquiry(self) -> List[Tuple[str, float]]:
        """Perform an inquiry of flashing letters to achieve RSVP.

        Returns:
            List[Tuple[str, float]]: List of timings of stimuli presented in the inquiry.
        """
        timing: List[Tuple[str, float]] = []
        self.preview_accepted = True

        if self.first_run:
            self._trigger_pulse()

        inquiry = self._generate_inquiry()

        for idx, stim_props in enumerate(inquiry):
            if idx == 0 and callable(self.first_stim_callback):
                self.first_stim_callback(stim_props['sti'])

            if self.preview_enabled and idx == self.preview_index:
                self.preview_accepted = self.preview_inquiry(timing)
            if not self.preview_accepted:
                break

            self.window.callOnFlip(
                self.trigger_callback.callback,
                self.experiment_clock,
                stim_props['sti_label'])

            stim_props['sti'].draw()
            self.draw_static()
            self.window.flip()
            core.wait(stim_props['time_to_present'])

            timing.append(self.trigger_callback.timing)
            self.trigger_callback.reset()

        self.draw_static()
        self.window.flip()

        return timing

    def _trigger_pulse(self) -> None:
        """Send a calibration trigger pulse.

        Uses a calibration trigger to determine any functional offsets needed for
        operation with this display. Sets first_stim_time and searches for the same
        stimuli output to the marker stream to reconcile timing offsets.
        """
        calibration_time = _calibration_trigger(
            self.experiment_clock,
            trigger_type=self.trigger_type,
            display=self.window)

        if not self.first_stim_time:
            self.first_stim_time = calibration_time[-1]
            self.first_run = False

    def preview_inquiry(self, timing: List[Tuple[str, float]]) -> bool:
        """Preview the inquiry before presentation.

        Given an inquiry defined to be presented via do_inquiry(), present the full inquiry
        to the user and allow input on whether the intended letter is present or not before
        going through the rapid serial visual presentation.

        Args:
            timing (List[Tuple[str, float]]): List to which timing information should be appended.

        Returns:
            bool: True if inquiry should be presented, False if a new one should be generated.

        Raises:
            AssertionError: If preview is not enabled or button handler is not initialized.
        """
        assert self.preview_enabled, "Preview feature not enabled."
        assert self.preview_button_handler, "Button handler must be initialized"
        assert self.preview_params is not None, "Preview parameters must be set"

        handler = self.preview_button_handler
        self.window.callOnFlip(
            self.trigger_callback.callback,
            self.experiment_clock,
            'inquiry_preview')
        self.draw_preview()

        handler.await_response()
        timing.append(self.trigger_callback.timing)
        if handler.has_response():
            timing.append((handler.response_label, handler.response_timestamp))

        self.trigger_callback.reset()
        self.draw_static()
        self.window.flip()
        core.wait(self.preview_params.preview_inquiry_isi)

        return handler.accept_result()

    def draw_preview(self) -> None:
        """Generate and draw the inquiry preview."""
        content = self._generate_inquiry_preview()
        content.draw()
        self.draw_static()
        self.window.flip()

    def _generate_inquiry_preview(self) -> visual.TextBox2:
        """Generate the inquiry preview box.

        Using the self.stimuli_inquiry list, construct a preview box to display to the user.
        Assumes the presence of a fixation (+).

        Returns:
            visual.TextBox2: The preview text box.

        Raises:
            AssertionError: If preview parameters are not set.
        """
        assert self.preview_params is not None, "Preview parameters must be set"
        text = ' '.join(self.stimuli_inquiry).split('+ ')[1]

        return self._create_stimulus(
            self.preview_params.preview_box_text_size,
            stimulus=text,
            units='height',
            stimuli_position=self.stimuli_pos,
            mode='textbox',
            align_text='left')

    def _generate_inquiry(self) -> List[Dict[str, Any]]:
        """Generate stimuli for next RSVP inquiry.

        Returns:
            List[Dict[str, Any]]: List of stimulus properties for the inquiry.
        """
        stim_info = []
        for idx, stim in enumerate(self.stimuli_inquiry):
            current_stim = {}

            current_stim['time_to_present'] = self.stimuli_timing[idx]

            this_stimuli_size = (self.size_list_sti[idx] if self.size_list_sti
                                 else self.stimuli_height)

            if stim.endswith('.png'):
                current_stim['sti'] = self._create_stimulus(
                    mode='image',
                    height=this_stimuli_size,
                    stimulus=stim
                )
                current_stim['sti'].size = resize_image(
                    current_stim['sti'].image, current_stim['sti'].win.size, this_stimuli_size)
                current_stim['sti_label'] = path.splitext(
                    path.basename(stim))[0]
            else:
                current_stim['sti'] = self._create_stimulus(
                    mode='text', height=this_stimuli_size)
                txt = stim
                current_stim['sti'].text = txt if txt != SPACE_CHAR else self.space_char
                current_stim['sti'].color = self.stimuli_colors[idx]
                current_stim['sti_label'] = txt

                text_width = current_stim['sti'].boundingBox[0]
                if text_width > self.window.size[0]:
                    screen_info = get_screen_info()
                    monitor_width = screen_info.width
                    monitor_height = screen_info.height
                    text_height = current_stim['sti'].boundingBox[1]
                    if self.window.size[0] == monitor_width:
                        new_text_width = monitor_width / monitor_height
                    else:
                        new_text_width = (
                            self.window.size[1] / monitor_height) * (
                                monitor_width / monitor_height)
                    new_text_height = (
                        text_height * new_text_width) / text_width
                    current_stim['sti'].height = new_text_height
            stim_info.append(current_stim)
        return stim_info

    def update_task_bar(self, text: Optional[str] = None) -> None:
        """Update task state.

        Args:
            text (Optional[str]): New text for task state.
        """
        if self.task_bar:
            self.task_bar.update(text)

    def wait_screen(self, message: str, message_color: str) -> None:
        """Display a wait screen with message and optional logo.

        Args:
            message (str): Message to be displayed while waiting.
            message_color (str): Color of the message to be displayed.

        Raises:
            Exception: If the logo image cannot be loaded.
        """
        wait_message = visual.TextStim(win=self.window,
                                       font=self.stimuli_font,
                                       text=message,
                                       height=.1,
                                       color=message_color,
                                       pos=(0, -.5),
                                       wrapWidth=2,
                                       colorSpace='rgb',
                                       opacity=1,
                                       depth=-6.0)

        try:
            wait_logo = visual.ImageStim(
                self.window,
                image=BCIPY_LOGO_PATH,
                pos=(0, .25),
                mask=None,
                ori=0.0)
            wait_logo.size = resize_image(
                BCIPY_LOGO_PATH,
                self.window.size,
                1)
            wait_logo.draw()

        except Exception as e:
            self.logger.exception(
                f'Cannot load logo image from path=[{BCIPY_LOGO_PATH}]')
            raise e

        wait_message.draw()
        self.window.flip()

    def _create_stimulus(
            self,
            height: float,
            mode: str = 'text',
            stimulus: str = '+',
            color: str = 'white',
            stimuli_position: Optional[Tuple[float, float]] = None,
            align_text: str = 'center',
            units: Optional[str] = None,
            wrap_width: Optional[float] = None,
            border: bool = False) -> Union[visual.TextStim, visual.ImageStim, visual.TextBox2]:
        """Create a stimulus object.

        Args:
            height (float): Height of the stimulus.
            mode (str, optional): Type of stimulus ('text', 'image', or 'textbox').
                Defaults to 'text'.
            stimulus (str, optional): Content of the stimulus. Defaults to '+'.
            color (str, optional): Color of the stimulus. Defaults to 'white'.
            stimuli_position (Optional[Tuple[float, float]], optional): Position of the stimulus.
                Defaults to None.
            align_text (str, optional): Text alignment. Defaults to 'center'.
            units (Optional[str], optional): Units for size/position. Defaults to None.
            wrap_width (Optional[float], optional): Width for text wrapping. Defaults to None.
            border (bool, optional): Whether to show border. Defaults to False.

        Returns:
            Union[visual.TextStim, visual.ImageStim, visual.TextBox2]: The created stimulus object.
        """
        if not stimuli_position:
            stimuli_position = self.stimuli_pos
        if mode == 'text':
            return visual.TextStim(
                win=self.window,
                color=color,
                height=height,
                text=stimulus,
                font=self.stimuli_font,
                pos=stimuli_position,
                wrapWidth=wrap_width,
                colorSpace='rgb',
                units=units,
                alignText=align_text,
                opacity=1,
                depth=-6.0)
        elif mode == 'image':
            return visual.ImageStim(
                win=self.window,
                image=stimulus,
                mask=None,
                units=units,
                pos=stimuli_position,
                size=(height, height),
                ori=0.0)
        elif mode == 'textbox':
            return visual.TextBox2(
                win=self.window,
                text=stimulus,
                color=color,
                colorSpace='rgb',
                borderWidth=2,
                borderColor='white',
                units=units,
                padding=0.05,
                font=self.textbox_font,
                letterHeight=height,
                size=[.55, .55],
                pos=stimuli_position,
                alignment=align_text,
                editable=False,
            )
        else:
            raise ValueError(
                f'RSVPDisplay asked to create a stimulus type=[{mode}] that is not supported.')
