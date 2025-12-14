"""Generates plots for replay session."""
import logging
from pathlib import Path
from typing import List, NamedTuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from bcipy.config import SESSION_DATA_FILENAME
from bcipy.core.session import evidence_records, read_session
from bcipy.simulator.util.artifact import TOP_LEVEL_LOGGER_NAME
from bcipy.simulator.util.metrics import session_paths

logger = logging.getLogger(TOP_LEVEL_LOGGER_NAME)


class Record(NamedTuple):
    """Record for plotting."""
    which_model: str
    response_value: float


def make_record(is_original_model: bool, is_target: bool,
                response_value: float) -> Record:
    """Construct a new Record"""
    prefix = "old" if is_original_model else "new"
    ev_type = "target" if is_target else "nontarget"

    return Record(f"{prefix}_{ev_type}", response_value)


def load_eeg_records(session_json: Path,
                     is_original_model: bool) -> List[Record]:
    """Load EEG evidence from session.json file for comparison with replay outputs.

    Parameters
    ----------
        session_json - path to session.json file
    """
    session = read_session(str(session_json))
    return [
        make_record(is_original_model=is_original_model,
                    is_target=bool(ev_record.is_target),
                    response_value=ev_record.eeg)
        for ev_record in evidence_records(session)
    ]


def prepare_data(sim_dir: str, data_folders: List[str]) -> List[Record]:
    """Prepare the data for plotting.

    Parameters
    ----------
        sim_dir - simulation directory path
        data_folders - data folders used in the simulation
    """

    original_sessions = [
        Path(folder, SESSION_DATA_FILENAME) for folder in data_folders
    ]
    simulated_sessions = session_paths(sim_dir)

    records = []
    for session_path in simulated_sessions:
        records.extend(load_eeg_records(session_path, is_original_model=False))
    for session_path in original_sessions:
        records.extend(load_eeg_records(session_path, is_original_model=True))

    return records


def plot(sim_dir: str, data: List[Record]) -> None:
    """Plot the summary

    Parameters
    ----------
        sim_dir - simulation directory; figures will be written here.
        df - dataframe with the evidence data
    """
    df = pd.DataFrame(data)
    logger.info(f"{df.describe()}")
    data_order = ["old_target", "new_target", "old_nontarget", "new_nontarget"]

    ax = sns.stripplot(
        x="which_model",
        y="response_value",
        data=df,
        order=data_order,
    )
    sns.boxplot(
        showmeans=True,
        meanline=True,
        meanprops={
            "color": "k",
            "ls": "-",
            "lw": 2
        },
        medianprops={"visible": False},
        whiskerprops={"visible": False},
        zorder=10,
        x="which_model",
        y="response_value",
        data=df,
        showfliers=False,
        showbox=False,
        showcaps=False,
        ax=ax,
        order=data_order,
    )

    ax.set(yscale="log")
    plt.savefig(Path(sim_dir, "response_values.stripplot.png"),
                dpi=150,
                bbox_inches="tight")
    plt.close()
    ax = sns.boxplot(
        x="which_model",
        y="response_value",
        data=df,
        order=data_order,
    )
    ax.set(yscale="log")
    plt.savefig(Path(sim_dir, "response_values.boxplot.png"),
                dpi=150,
                bbox_inches="tight")


def comparison_metrics(sim_dir, data_folders: List[str]) -> None:
    """Compare the simulated results with the original."""
    data = prepare_data(sim_dir, data_folders)
    plot(sim_dir, data)
