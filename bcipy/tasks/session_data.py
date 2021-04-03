"""Module for functionality related to session-related data."""
from typing import Dict, List
from collections import Counter


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

    @classmethod
    def from_dict(cls, data: dict):
        """Deserializes from a dict

        Parameters:
        ----------
            data - a dict in the format of the data output by the as_dict
                method.
        """
        return cls(**data)

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

    def stim_evidence(self, alphabet: List[str], n: int = 5) -> Dict:
        """Returns a dict of stim sequence data useful for debugging. Evidences
        are paired with the appropriate alphabet letter for easier visual
        scanning. Also, an additional attribute is provided to display the
        top n most likely letters based on the current evidence.

        Parameters:
        -----------
            alphabet - list of stim in the same order as the evidences.
            n - number of most likely elements to include
        """
        likelihood = dict(zip(alphabet, self.likelihood))
        return {
            'stimuli': self.stimuli,
            'lm_evidence': dict(zip(alphabet, self.lm_evidence)),
            'eeg_evidence': dict(zip(alphabet, self.eeg_evidence)),
            'likelihood': likelihood,
            'most_likely': dict(Counter(likelihood).most_common(5))
        }


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
        """Add another series unless the last one is empty"""
        if self.last_series():
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

    def as_dict(self, alphabet=None, evidence_only: bool = False) -> Dict:
        """Dict representation"""
        series_dict = {}
        for i, series in enumerate(self.series):
            if series:
                series_counter = str(i + 1)
                series_dict[series_counter] = {}
                for series_index, stim in enumerate(series):
                    if evidence_only and alphabet:
                        stim_dict = stim.stim_evidence(alphabet)
                    else:
                        stim_dict = stim.as_dict()
                    series_dict[series_counter][str(series_index)] = stim_dict

        return {
            'session': self.save_location,
            'session_type': self.session_type,
            'paradigm': self.paradigm,
            'series': series_dict,
            'total_time_spent': self.total_time_spent,
            'total_number_series': self.total_number_series
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Deserialize from a dict.

        Parameters:
        ----------
            data - a dict in the format of the data output by the as_dict
                method.
        """
        session = cls(save_location=data['session'],
                      session_type=data['session_type'],
                      paradigm=data['paradigm'])
        session.total_time_spent = data['total_time_spent']

        if data['series']:
            session.series.clear()
            for series_counter in sorted(data['series'].keys()):
                session.series.append([])
                for sequence_counter in sorted(data['series'][series_counter]):
                    sequence_dict = data['series'][series_counter][
                        sequence_counter]
                    session.add_sequence(StimSequence.from_dict(sequence_dict))

        return session
