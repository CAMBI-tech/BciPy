"""Summarize metrics across simulator runs."""

import logging
import sys
from collections import Counter
from json import load
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from bcipy.config import DEFAULT_ENCODING
from bcipy.helpers.acquisition import max_inquiry_duration
from bcipy.io.load import load_json_parameters
from bcipy.io.save import save_json_data
from bcipy.simulator.util.artifact import TOP_LEVEL_LOGGER_NAME

logger = logging.getLogger(TOP_LEVEL_LOGGER_NAME)

SUMMARY_DATA_FILE_NAME = "summary_data.json"
TASK_SUMMARY_PREFIX = "task_summary__"


def add_item(container: Dict[str, List], key: str, value: Any):
    """Add or append value"""
    if key in container:
        container[key].append(value)
    else:
        container[key] = [value]


def get_final_typed(session_dict: Dict) -> str:
    """Get the final typed word"""
    all_series = session_dict['series']
    if all_series.keys():
        last_series_key = str(max(map(int, all_series.keys())))
        inquiries = all_series.get(last_series_key, {})
        if inquiries.keys():
            last_inquiry_key = str(max(map(int, inquiries.keys())))
            return inquiries.get(last_inquiry_key).get('next_display_state',
                                                       '')
    return ''


def calculate_duration(inquiry_count: int, inquiry_seconds: float) -> float:
    """Compute the total number of seconds."""
    return inquiry_count * inquiry_seconds


def add_items(combined_metrics: Dict, session_dict: Dict,
              inquiry_seconds: float):
    """Add all items of interest from the session data."""
    keys = [
        'total_number_series', 'total_inquiries', 'total_selections',
        'inquiries_per_selection'
    ]
    for key in keys:
        add_item(combined_metrics, key, session_dict[key])

    for key in session_dict['task_summary'].keys():
        if 'switch' not in key and 'rate' not in key:
            add_item(combined_metrics, f"{TASK_SUMMARY_PREFIX}{key}",
                     session_dict['task_summary'][key])

    add_item(combined_metrics, 'typed', get_final_typed(session_dict))
    add_item(
        combined_metrics, 'total_seconds',
        calculate_duration(session_dict['total_inquiries'], inquiry_seconds))


def session_paths(sim_dir: str) -> List[Path]:
    """List of paths to the session.json files"""
    return [
        Path(run_dir, "session.json")
        for run_dir in sorted(Path(sim_dir).glob("run*/"))
    ]


def sim_parameters(sim_dir: str) -> Dict[str, Any]:
    """Simulation parameters"""
    return load_json_parameters(str(Path(sim_dir, "parameters.json")), True)


def summarize(sim_dir: str) -> Dict[str, List[Any]]:
    """Summarize all session runs."""
    inquiry_seconds = max_inquiry_duration(sim_parameters(sim_dir))
    combined_metrics: Dict[str, List[Any]] = {}
    for session_path in session_paths(sim_dir):
        with open(session_path, 'r', encoding=DEFAULT_ENCODING) as json_file:
            session_dict = load(json_file)
            add_items(combined_metrics, session_dict, inquiry_seconds)
    return combined_metrics


def rename_df_column(colname: str) -> str:
    """Rename dataframe columns for reporting"""
    if colname.startswith(TASK_SUMMARY_PREFIX):
        return colname[len(TASK_SUMMARY_PREFIX):]
    return colname


def log_descriptive_stats(df: pd.DataFrame) -> None:
    """Log the descriptive statistics for the given dataframe.

    See https://stackoverflow.com/questions/25351968/how-can-i-display-full-non-truncated-dataframe-information-in-html-when-conver
    """
    df.rename(columns=rename_df_column, inplace=True)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', "{:20,.2f}".format)
    pd.set_option('display.max_colwidth', None)

    logger.info(f"Metrics:\n{df.describe()}")

    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')


def plot_save_path(sim_dir: str) -> str:
    """Save path for plots."""
    return f"{sim_dir}/metrics.png"


def plot_results(df: pd.DataFrame,
                 save_path: Optional[str] = None,
                 show: bool = True) -> None:
    """Plot the dataframe"""
    cols = [
        'total_selections', 'total_inquiries', 'inquiries_per_selection',
        'selections_correct', 'selections_incorrect'
    ]
    df.boxplot(column=cols, figsize=(11, 8.5))
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


def report(sim_dir: str, show_plots: bool = False) -> None:
    """Summarize the data, write as a JSON file, and output a summary to
    the top level log file."""
    summary = summarize(sim_dir)
    save_json_data(summarize(sim_dir), sim_dir, SUMMARY_DATA_FILE_NAME)

    df = pd.DataFrame(summary)
    log_descriptive_stats(df)

    logger.info("Typed:")
    logger.info(Counter(summary['typed']))
    ave_minutes = round((df['total_seconds'] / 60).mean(), 2)
    logger.info(f"Average duration: {ave_minutes} minutes")
    plot_results(df, save_path=plot_save_path(sim_dir), show=show_plots)


if __name__ == '__main__':
    report(sim_dir=sys.argv[1])
