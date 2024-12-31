from bcipy.display.paradigm.vep.copy_phrase_display import VEPDisplay
from bcipy.helpers.symbols import SPACE_CHAR


class CopyPhraseDisplayVEP(VEPDisplay):
    """Calibration Display."""

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
