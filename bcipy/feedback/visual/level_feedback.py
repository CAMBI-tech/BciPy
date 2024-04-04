from typing import List, NamedTuple

from psychopy import core, visual
from psychopy.visual.rect import Rect

from bcipy.display import init_display_window
from bcipy.display.components.layout import (centered, left_of, right_of,
                                             scaled_size)
from bcipy.feedback.feedback import Feedback
from bcipy.helpers.clock import Clock
from bcipy.helpers.load import load_json_parameters
from bcipy.helpers.parameters import Parameters


class BoxConfig(NamedTuple):
    """Configuration for the box elements that make up the display."""
    height: float = 0.35
    line_color: str = 'white'
    target_line_color: str = 'white'
    colors: List[str] = ['red', 'darkorange', 'yellow', 'lightgreen', 'green']
    line_width: int = 2
    target_line_width: int = 6
    nontarget_opacity: float = 0.7


def construct_bar_indicator(win: visual.Window,
                            config: BoxConfig) -> List[Rect]:
    """Construct the stimuli."""
    layout = centered()
    num_boxes = len(config.colors)

    box_size = scaled_size(config.height, win.size)
    box_width = box_size[0]

    center_x, center_y = layout.center
    start_pos_x = left_of(center_x, int(num_boxes / 2) * box_width)

    # Account for an even number of boxes
    if (len(config.colors) % 2 == 0):
        start_pos_x = right_of(start_pos_x, box_width / 2)

    positions = [(right_of(start_pos_x, (i * box_width)), center_y)
                 for i in range(num_boxes)]

    return [
        Rect(win,
             pos=position,
             size=box_size,
             fillColor=color,
             lineColor=config.line_color,
             lineWidth=config.line_width,
             opacity=0.7,
             units='norm') for position, color in zip(positions, config.colors)
    ]


class LevelFeedback(Feedback):
    """Level Feedback.

    A progress bar like feedback to indicate current abilities in a BCI task. This could
        be attentiveness, muscle or eye activity, etc.

    It does not return stimuli timing or allow for parameterized configuration of levels / color gradient
    at this time.
    """

    def __init__(self, display: visual.Window, parameters: Parameters,
                 clock: Clock):
        super().__init__(feedback_type='visual')
        self.display = display
        self.clock = clock
        self.presentation_time = 2  # seconds
        self.config = BoxConfig()
        self.stimuli = construct_bar_indicator(display, self.config)

    def configure(self):
        """Override"""

    def administer(self, position: int = 1):
        """Administer the feedback"""
        assert 1 <= position <= 5, "Position must be between 1 and 5"

        # indexing starts at zero but position does not ;)
        selected_index = position - 1

        # loop through the levels wanted and construct the stimuli
        for index, rect in enumerate(self.stimuli):
            # this is the target indicator level and should stand out of the
            # rest
            if selected_index == index:
                rect.lineWidth = self.config.target_line_width
                rect.opacity = 1.0
            else:
                rect.lineWidth = self.config.line_width
                rect.opacity = self.config.nontarget_opacity
            rect.draw()

        # flip the display and present for a time
        time = ['visual_feedback', self.clock.getTime()]
        self.display.flip()
        core.wait(self.presentation_time)

        return time


def main():
    """Main method"""
    parameters = load_json_parameters('bcipy/parameters/parameters.json',
                                      value_cast=True)
    display = init_display_window(parameters)
    clock = core.Clock()
    feedback = LevelFeedback(display, parameters, clock)

    for position in range(1, len(feedback.config.colors) + 1):
        feedback.administer(position)


if __name__ == '__main__':
    main()
