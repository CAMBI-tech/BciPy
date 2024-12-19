"""Functions for trial-related data."""
import itertools as it
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional, Tuple

import numpy as np

from bcipy.config import SESSION_DATA_FILENAME
from bcipy.core.list import find_index
from bcipy.core.session import read_session
from bcipy.simulator.data.data_process import ExtractedExperimentData


class Trial(NamedTuple):
    """Data for a given trial (a symbol within an Inquiry).

    Attrs
    -----
        source - directory of the data source
        series - starts at 1; computed from session.json
        series_inquiry - starts at 0; inquiry within the series; computed
            from session.json
        inquiry_n - starts at 0; value from the start of the session (does not
            reset at each series); can be computed from raw_data.
        inquiry_pos  - starts at 1; position in which the symbol was presented
        symbol - alphabet symbol that was presented
        target - 1 or 0 indicating a boolean of whether this was a target symbol
        eeg - EEG data associated with this trial
    """
    source: str
    series: Optional[int]
    series_inquiry: Optional[int]
    inquiry_n: int
    inquiry_pos: int
    symbol: str
    target: int
    eeg: np.ndarray  # Channels by Samples ; ndarray.shape = (channel_n, sample_n)

    def __str__(self):
        fields = [
            f"source='{self.source}'", f"series={self.series}",
            f"series_inquiry={self.series_inquiry}",
            f"inquiry_n={self.inquiry_n}", f"inquiry_pos={self.inquiry_pos}",
            f"symbol='{self.symbol}'", f"target={self.target}",
            f"eeg={self.eeg.shape}"
        ]
        return f"Trial({', '.join(fields)})"

    def __repr__(self):
        return str(self)


def session_series_counts(source_dir: str) -> List[int]:
    """Read the session.json file in the provided directory
    and compute the number of inquiries per series."""
    session_path = Path(source_dir, SESSION_DATA_FILENAME)
    if session_path.exists():
        session = read_session(str(session_path))
        return [len(series) for series in session.series]
    return []


def series_inquiry(series_counts: List[int],
                   inquiry_n: int) -> Tuple[Optional[int], Optional[int]]:
    """Given the number of inquiries per series and the inquiry from the start of the session,
    compute the series and inquiry relative to the start of that series.

    Parameters
    ----------
        series_counts - number of inquiries for each series in the session.
        inquiry_n - index of the inquiry from the start of the session
            starts at 0.
    Returns
    -------
        a tuple of the series (1-based), inquiry_index (0-based) relative to series.
    """
    if series_counts:
        accumulations = list(it.accumulate(series_counts))
        series_index = find_index(accumulations,
                                  match_item=lambda val: val > inquiry_n)
        if series_index is not None:
            if series_index > 0:
                inq_index = inquiry_n - accumulations[series_index - 1]
            else:
                inq_index = inquiry_n
            return (series_index + 1, inq_index)
    return (None, None)


def convert_trials(
    data_source: ExtractedExperimentData,
    get_series_counts: Callable[[str], List[int]] = session_series_counts
) -> List[Trial]:
    """Convert extracted data from a single data source to a list of Trials.

    Parameters
    ----------
        data_source - Data from an acquisition device after reshaping and filtering.
        get_series_counts - function to get the session metadata needed to assign
            series and relative_inquiry information to each trial. The function should
            take the source_directory path as an input and return the number of inquiries
            in each series. By default this is read from the session.json in the source_dir.
    """
    trials = []
    symbols_by_inquiry = data_source.symbols_by_inquiry
    labels_by_inquiry = data_source.labels_by_inquiry
    series_counts = get_series_counts(data_source.source_dir)

    for i, inquiry_eeg in enumerate(data_source.trials_by_inquiry):
        # iterate through each inquiry
        inquiry_symbols = symbols_by_inquiry[i]
        inquiry_labels = labels_by_inquiry[i]

        for sym_i, symbol in enumerate(inquiry_symbols):
            # iterate through each symbol in the inquiry
            series, inquiry = series_inquiry(series_counts, i)
            eeg_samples = [channel[sym_i]
                           for channel in inquiry_eeg]  # (channel_n, sample_n)
            trials.append(
                Trial(source=data_source.source_dir,
                      series=series,
                      series_inquiry=inquiry,
                      inquiry_n=i,
                      inquiry_pos=sym_i + 1,
                      symbol=symbol,
                      target=inquiry_labels[sym_i],
                      eeg=np.array(eeg_samples)))
    return trials
