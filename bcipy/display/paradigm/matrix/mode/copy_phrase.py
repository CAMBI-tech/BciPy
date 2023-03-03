"""Matrix Copy Phrase Display"""
from typing import List, Optional

from bcipy.display.components.task_bar import CopyPhraseTaskBar, TaskBar
from bcipy.display.paradigm.matrix.display import (InformationProperties,
                                                   MatrixDisplay,
                                                   StimuliProperties,
                                                   TaskDisplayProperties, core,
                                                   visual)


class MatrixCopyPhraseDisplay(MatrixDisplay):
    """Copy Phrase Display."""

    def __init__(self,
                 window: visual.Window,
                 experiment_clock: core.Clock,
                 stimuli: StimuliProperties,
                 task_bar_config: TaskDisplayProperties,
                 info: InformationProperties,
                 trigger_type: str = 'text',
                 symbol_set: Optional[List[str]] = None,
                 starting_spelled_text: str = ''):
        self.starting_spelled_text = starting_spelled_text
        super().__init__(window,
                         experiment_clock,
                         stimuli,
                         task_bar_config,
                         info,
                         trigger_type,
                         symbol_set,
                         should_prompt_target=True)

    def build_task_bar(self) -> TaskBar:
        """Creates a TaskBar"""
        return CopyPhraseTaskBar(self.window,
                                 self.task_bar_config,
                                 spelled_text=self.starting_spelled_text)
