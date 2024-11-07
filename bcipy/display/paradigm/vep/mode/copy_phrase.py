from psychopy import visual
from bcipy.display.paradigm.rsvp.display import VEPDisplay

"""Note:

"""


class CopyPhraseDisplayVEP(VEPDisplay):
    """ Copy Phrase display object of VEP

        Custom attributes:
            None
    """

    def __init__(
            self,
            window: visual.Window,
            experiment_clock: Clock,
            stimuli: VEPStimuliProperties,
            task_bar: TaskBar,
            info: InformationProperties,
            box_config: BoxConfiguration,
            trigger_type: str = 'text',
            symbol_set: Optional[List[str]] = None,
            flicker_rates: List[int] = DEFAULT_FLICKER_RATES,
            should_prompt_target: bool = True,
            frame_rate: Optional[float] = None):
        """ Initializes Copy Phrase Task Objects """

        super().__init__(window,
                         clock,
                         experiment_clock,
                         stimuli,
                         task_bar,
                         info,
                         trigger_type=trigger_type,
                         symbol_set=symbol_set,
                         flicker_rates=flicker_rates,
                         should_prompt_target=should_prompt_target,
                         frame_rate=frame_rate)
