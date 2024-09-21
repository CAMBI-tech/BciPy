from typing import List

from bcipy.simulator.helpers.data_engine import Trial


def format_samples(sample_rows: List[Trial]) -> str:
    """Returns a tabular representation of the sample rows."""
    return '\n'.join([str(row) for row in sample_rows])
