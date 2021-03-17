"""Module for functionality related to session-related data."""
from typing import Dict, List


class StimSequence:
    """Represents a sequence of stimuli."""

    def __init__(self,
                 stimuli: List[str],
                 eeg_len: int,
                 timing_sti: List[float],
                 triggers: List[List],
                 target_info: List[str],
                 target_letter: str,
                 current_text: str,
                 copy_phrase: str,
                 next_display_state: str = None,
                 lm_evidence: List[float] = None,
                 eeg_evidence: List[float] = None,
                 likelihood: List[float] = None):
        super().__init__()
        self.stimuli = stimuli
        self.eeg_len = eeg_len
        self.timing_sti = timing_sti
        self.triggers = triggers
        self.target_info = target_info
        self.target_letter = target_letter
        self.current_text = current_text
        self.copy_phrase = copy_phrase
        self.next_display_state = next_display_state

        # TODO: refactor for multimodal; List of Evidences?
        self.lm_evidence = lm_evidence or []
        self.eeg_evidence = eeg_evidence or []
        self.likelihood = likelihood or []

    def as_dict(self) -> Dict:
        data = {
            'stimuli': self.stimuli,
            'eeg_len': self.eeg_len,
            'timing_sti': self.timing_sti,
            'triggers': self.triggers,
            'target_info': self.target_info,
            'target_letter': self.target_letter,
            'current_text': self.current_text,
            'copy_phrase': self.copy_phrase,
            'next_display_state': self.next_display_state
        }

        if self.lm_evidence:
            data['lm_evidence'] = self.lm_evidence
        if self.eeg_evidence:
            data['eeg_evidence'] = self.eeg_evidence
        if self.likelihood:
            data['likelihood'] = self.likelihood
        return data


class Session:
    """Represents a data collection session. Not all tasks record session data."""

    def __init__(self,
                 save_location: str,
                 session_type: str = 'Copy Phrase',
                 paradigm: str = 'RSVP'):
        super().__init__()
        self.save_location = save_location
        self.session_type = session_type
        self.paradigm = paradigm
        self.series: List[List[StimSequence]] = [[]]
        self.total_time_spent = 0

    @property
    def total_number_series(self) -> int:
        """Total number of series that contain sequences."""
        return len([lst for lst in self.series if lst])

    def add_series(self):
        if len(self.last_series()) > 0:
            self.series.append([])

    def add_sequence(self,
                     stim_sequence: StimSequence,
                     new_series: bool = False):
        """Append sequence information

        Parameters:
        -----------
            stim_sequence - data to append
            new_series - a True value indicates that this is the first stim of
                a new series.
        """
        if new_series:
            self.add_series()
        self.last_series().append(stim_sequence)

    def last_series(self) -> List[StimSequence]:
        """Returns the last series"""
        return self.series[-1]

    def as_dict(self) -> Dict:
        """Dict representation"""
        series_dict = {}
        for i, series in enumerate(self.series):
            if series:
                series_counter = str(i + 1)
                series_dict[series_counter] = {}
                for series_index, stim in enumerate(series):
                    series_dict[series_counter][str(
                        series_index)] = stim.as_dict()

        return {
            'session': self.save_location,
            'session_type': self.session_type,
            'paradigm': self.paradigm,
            'series': series_dict,
            'total_time_spent': self.total_time_spent,
            'total_number_series': self.total_number_series
        }
