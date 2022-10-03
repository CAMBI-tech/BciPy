"""Module for functionality related to session-related data."""
from collections import Counter
from enum import Enum
from typing import Any, Dict, List, Optional

EVIDENCE_SUFFIX = "_evidence"


def rounded(values: List[float], precision: int) -> List[float]:
    """Round the list of values to the given precision.

    Parameters
    ----------
        values - values to round
    """
    return [round(value, precision) for value in values]


class EvidenceType(Enum):
    """Enum of the supported evidence types used in the various spelling tasks."""
    LM = 'LM'  # Language Model
    ERP = 'ERP'  # Event-Related Potential using EEG signals
    BTN = 'BTN'  # Button

    @classmethod
    def list(cls) -> List[str]:
        """List of evidence types"""
        return [ev_type.name for ev_type in cls]

    @classmethod
    def deserialized(cls, serialized_name: str) -> 'EvidenceType':
        """Deserialized name of the given evidence type.
        Parameters:
            evidence_name - ex. 'lm_evidence'
        Returns:
            deserialized value: ex. EvidenceType.LM
        """
        if serialized_name == 'eeg_evidence':
            return EvidenceType.ERP
        return cls(serialized_name[:-len(EVIDENCE_SUFFIX)].upper())

    def __str__(self) -> str:
        return self.name

    @property
    def serialized(self) -> str:
        """Name used when serialized to a json file."""
        if self == EvidenceType.ERP:
            return 'eeg_evidence'
        return f'{self.name.lower()}{EVIDENCE_SUFFIX}'


class Inquiry:
    """Represents a sequence of stimuli.

    Parameters:
    ----------
        stimuli - list of stimuli presented (letters, icons, etc)
        timing - duration in seconds for each stimulus
        target_info - targetness ('nontarget', 'target', etc) for each stimulus
        target_letter - current letter that the user is attempting to spell
        current_text - letters spelled so far
        target_text - word or words the user is attempting to spell
        next_display_state - text to be displayed after evaluating the current evidence
        lm_evidence - language model evidence for each stimulus
        eeg_evidence - eeg evidence for each stimulus
        likelihood - combined likelihood for each stimulus
    """

    def __init__(self,
                 stimuli: List[str],
                 timing: List[float],
                 triggers: List[List],
                 target_info: List[str],
                 target_letter: str = None,
                 current_text: str = None,
                 target_text: str = None,
                 selection: str = None,
                 next_display_state: str = None,
                 likelihood: List[float] = None):
        super().__init__()
        self.stimuli = stimuli
        self.timing = timing
        self.triggers = triggers
        self.target_info = target_info
        self.target_letter = target_letter
        self.current_text = current_text
        self.target_text = target_text
        self.selection = selection
        self.next_display_state = next_display_state

        self.evidences: Dict[EvidenceType, List[float]] = {}
        self.likelihood = likelihood or []
        # Precision used for serialization of evidence values.
        self.precision = None

    @property
    def lm_evidence(self):
        """Language model evidence"""
        return self.evidences.get(EvidenceType.LM, [])

    @property
    def eeg_evidence(self):
        """EEG evidence"""
        return self.evidences.get(EvidenceType.ERP, [])

    @property
    def decision_made(self) -> bool:
        """Returns true if the result of the inquiry was a decision."""
        return self.current_text != self.next_display_state

    @property
    def is_correct_decision(self) -> bool:
        """Indicates whether the current selection was the target"""
        return self.selection and (self.selection == self.target_letter)

    @classmethod
    def from_dict(cls, data: dict):
        """Deserializes from a dict

        Parameters:
        ----------
            data - a dict in the format of the data output by the as_dict
                method.
        """
        # partition into evidence data and other data.

        evidences = {
            EvidenceType.deserialized(name): value
            for name, value in data.items() if name.endswith(EVIDENCE_SUFFIX)
        }

        non_evidence_data = {
            name: value
            for name, value in data.items()
            if not name.endswith(EVIDENCE_SUFFIX)
        }
        inquiry = cls(**non_evidence_data)
        if len(data['stimuli']) == 1 and isinstance(data['stimuli'][0], list):
            # flatten
            inquiry.stimuli = data['stimuli'][0]
        inquiry.evidences = evidences
        return inquiry

    def as_dict(self) -> Dict:
        """Dict representation"""
        data = {
            'stimuli': self.stimuli,
            'timing': self.timing,
            'triggers': self.triggers,
            'target_info': self.target_info,
            'target_letter': self.target_letter,
            'current_text': self.current_text,
            'target_text': self.target_text,
            'selection': self.selection,
            'next_display_state': self.next_display_state
        }

        for evidence_type, evidence in self.evidences.items():
            data[evidence_type.serialized] = self.format(evidence)

        if self.likelihood:
            data['likelihood'] = self.format(self.likelihood)
        return data

    def stim_evidence(self,
                      symbol_set: List[str],
                      n_most_likely: int = 5) -> Dict[str, Any]:
        """Returns a dict of stim sequence data useful for debugging. Evidences
        are paired with the appropriate symbol for easier visual
        scanning. Also, an additional attribute is provided to display the
        top n most likely symbols based on the current evidence.

        Parameters:
        -----------
            symbol_set - list of stim in the same order as the evidences.
            n_most_likely - number of most likely elements to include
        """
        likelihood = dict(zip(symbol_set, self.format(self.likelihood)))
        data: Dict[str, Any] = {
            'stimuli': self.stimuli,
        }
        for evidence_type, evidence in self.evidences.items():
            data[evidence_type.serialized] = dict(
                zip(symbol_set, self.format(evidence)))
        data['likelihood'] = likelihood
        data['most_likely'] = dict(
            Counter(likelihood).most_common(n_most_likely))
        return data

    def format(self, evidence: List[float]) -> List[float]:
        """Format the evidence for output.

        Parameters
        ----------
            evidence - list of evidence values
        """
        if self.precision:
            return rounded(evidence, self.precision)
        return evidence


class Session:
    """Represents a data collection session. Not all tasks record session data."""

    def __init__(self,
                 save_location: str,
                 task: str = 'Copy Phrase',
                 mode: str = 'RSVP',
                 symbol_set: List[str] = None,
                 decision_threshold: float = None):
        super().__init__()
        self.save_location = save_location
        self.task = task
        self.mode = mode
        self.series: List[List[Inquiry]] = [[]]
        self.total_time_spent = 0
        self.time_spent_precision = 2
        self.task_summary = {}
        self.symbol_set = symbol_set
        self.decision_threshold = decision_threshold

    @property
    def total_number_series(self) -> int:
        """Total number of series that contain sequences."""
        return len([lst for lst in self.series if lst])

    @property
    def total_number_decisions(self) -> int:
        """Total number of series that ended in a decision."""
        # An alternate implementation would be to count the inquiries with
        # decision_made property of true.
        return len(self.series) - 1

    @property
    def total_inquiries(self) -> int:
        """Total number of inquiries presented."""
        return sum([len(lst) for lst in self.series])

    @property
    def inquiries_per_selection(self) -> Optional[float]:
        """Inquiries per selection"""
        selections = self.total_number_decisions
        if selections == 0:
            return None
        return self.total_inquiries / selections

    @property
    def all_inquiries(self) -> List[Inquiry]:
        """List of all Inquiries for the whole session"""
        return [inq for inquiries in self.series for inq in inquiries if inquiries]

    def has_evidence(self) -> bool:
        """Tests whether any inquiries have evidence."""
        return any(inq.evidences for inq in self.all_inquiries)

    def add_series(self):
        """Add another series unless the last one is empty"""
        if self.last_series():
            self.series.append([])

    def add_sequence(self, inquiry: Inquiry, new_series: bool = False):
        """Append sequence information

        Parameters:
        -----------
            inquiry - data to append
            new_series - a True value indicates that this is the first stim of
                a new series.
        """
        if new_series:
            self.add_series()
        self.last_series().append(inquiry)

    def last_series(self) -> List[Inquiry]:
        """Returns the last series"""
        return self.series[-1]

    def last_inquiry(self) -> Optional[Inquiry]:
        """Returns the last inquiry of the last series."""
        series = self.last_series()
        if series:
            return series[-1]
        return None

    def latest_series_is_empty(self) -> bool:
        """Whether the latest series has had any inquiries added to it."""
        return len(self.last_series()) == 0

    def as_dict(self,
                evidence_only: bool = False) -> Dict[str, Any]:
        """Dict representation"""
        series_dict: Dict[str, Any] = {}
        for i, series in enumerate(self.series):
            if series:
                series_counter = str(i + 1)
                series_dict[series_counter] = {}
                for series_index, stim in enumerate(series):
                    if evidence_only and self.symbol_set:
                        stim_dict = stim.stim_evidence(self.symbol_set)
                    else:
                        stim_dict = stim.as_dict()
                    series_dict[series_counter][str(series_index)] = stim_dict

        info = {
            'session': self.save_location,
            'task': self.task,
            'mode': self.mode,
            'symbol_set': self.symbol_set,
            'decision_threshold': self.decision_threshold,
            'series': series_dict,
            'total_time_spent': round(self.total_time_spent,
                                      self.time_spent_precision),
            'total_minutes': round(self.total_time_spent / 60, self.time_spent_precision),
            'total_number_series': self.total_number_series,
            'total_inquiries': self.total_inquiries,
            'total_selections': self.total_number_decisions,
            'inquiries_per_selection': self.inquiries_per_selection
        }

        if self.task_summary:
            info['task_summary'] = self.task_summary
        return info

    @classmethod
    def from_dict(cls, data: dict):
        """Deserialize from a dict.

        Parameters:
        ----------
            data - a dict in the format of the data output by the as_dict
                method.
        """
        session = cls(save_location=data['session'],
                      task=data['task'],
                      mode=data['mode'],
                      symbol_set=data['symbol_set'],
                      decision_threshold=data['decision_threshold'])
        session.total_time_spent = data['total_time_spent']

        if data['series']:
            session.series.clear()

            for series_key in sorted(data['series'].keys(), key=int):
                session.series.append([])

                for inquiry_key in sorted(data['series'][series_key], key=int):
                    inquiry_dict = data['series'][series_key][inquiry_key]
                    session.add_sequence(Inquiry.from_dict(inquiry_dict))

        return session
