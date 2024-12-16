"""Summarize metrics across simulator runs."""

import logging
from collections import Counter
from json import load
from pathlib import Path
from typing import Dict, List

import pandas as pd

from bcipy.config import DEFAULT_ENCODING
from bcipy.io.save import save_json_data
from bcipy.simulator.util.artifact import TOP_LEVEL_LOGGER_NAME

logger = logging.getLogger(TOP_LEVEL_LOGGER_NAME)

SUMMARY_DATA_FILE_NAME = "summary_data.json"
TASK_SUMMARY_PREFIX = "task_summary__"


def add_item(container: Dict[str, List], key: str, value: float):
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


def add_items(combined_metrics: Dict, session_dict: Dict):
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


def session_paths(sim_dir: str) -> List[Path]:
    """List of paths to the session.json files"""
    return [
        Path(run_dir, "session.json")
        for run_dir in sorted(Path(sim_dir).glob("run*/"))
    ]


def summarize(sim_dir: str) -> Dict:
    """Summarize all session runs."""
    combined_metrics = {}
    for session_path in session_paths(sim_dir):
        with open(session_path, 'r', encoding=DEFAULT_ENCODING) as json_file:
            print("Loading: " + session_path)
            session_dict = load(json_file)
            add_items(combined_metrics, session_dict)
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


def report(sim_dir: str) -> None:
    """Summarize the data, write as a JSON file, and output a summary to
    the top level log file."""
    summary = summarize(sim_dir)
    save_json_data(summarize(sim_dir), sim_dir, SUMMARY_DATA_FILE_NAME)

    df = pd.DataFrame(summary)
    log_descriptive_stats(df)

    logger.info("Typed:")
    logger.info(Counter(summary['typed']))
