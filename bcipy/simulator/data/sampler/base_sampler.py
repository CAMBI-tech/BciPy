from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

import numpy as np

from bcipy.simulator.data.data_engine import RawDataEngine
from bcipy.simulator.data.trial import Trial
from bcipy.simulator.util.state import SimState


def default_reshaper(eeg_responses: List[np.ndarray]) -> np.ndarray:
    """Default data reshaper.

    Returns
    -------
    ndarray with shape (channel_n, trial_n, sample_n)
    """

    channels_eeg: List[List[np.ndarray]] = [
        [] for i in range(len(eeg_responses[0]))
    ]

    for _i, trial_channels_eeg in enumerate(eeg_responses):
        for c_i, channel_eeg in enumerate(trial_channels_eeg):
            channels_eeg[c_i].append(channel_eeg)

    return np.array(channels_eeg)


def format_samples(sample_rows: List[Trial]) -> str:
    """Returns a tabular representation of the sample rows."""
    return '\n'.join([str(row) for row in sample_rows])


class Sampler(ABC):
    """Represents a strategy for sampling signal model data from a DataEngine
    comprised of signal data from one or more data collection sessions.
    """

    def __init__(self, data_engine: RawDataEngine):
        self.data_engine: RawDataEngine = data_engine
        self.model_input_reshaper: Callable = default_reshaper

    @abstractmethod
    def sample(self, state: SimState) -> List[Trial]:
        """
        Query the data engine for a list of trials corresponding to each
        currently displayed symbol.

        Parameters
        ----------
            state - specifies the target symbol and current inquiry (displayed symbols).

        Returns
        -------
            a list of Trials with an item for each symbol in the current inquiry.
        """
        raise NotImplementedError

    def sample_data(self, state: SimState) -> np.ndarray:
        """
        Query for trials and reshape for signal model input according to the
        provided reshaper.

        Return:
            ndarray of shape (n_channel, n_trial, n_sample)
        """
        trials = self.sample(state)
        return self.reshaped(trials)

    def sample_with_context(self,
                            state: SimState) -> Tuple[np.ndarray, List[Trial]]:
        """
        Returns
        -------
            A tuple of the reshaped data (ndarray of shape (n_channel, n_trial, n_sample)) as
            well as a list of Trial data (metadata and data not reshaped) for context.
        """
        trials = self.sample(state)
        data = self.reshaped(trials)
        return data, trials

    def reshaped(self, sample_rows: List[Trial]) -> np.ndarray:
        """Returns the queried trials reshaped into a format that a model can predict."""
        return self.model_input_reshaper([trial.eeg for trial in sample_rows])

    def set_reshaper(self, reshaper: Callable):
        """Set the reshaper"""
        self.model_input_reshaper = reshaper

    def __str__(self):
        return f"<{self.__class__.__name__}>"
