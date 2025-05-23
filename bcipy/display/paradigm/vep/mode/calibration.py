from bcipy.display.paradigm.vep.display import VEPDisplay
from bcipy.helpers.symbols import SPACE_CHAR


class CalibrationDisplayVEP(VEPDisplay):
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
            calibration_mode: bool = True,
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
                         calibration_mode=calibration_mode,
                         frame_rate=frame_rate)
