import logging as logger
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bcipy.helpers.parameters import Parameters

logger.getLogger().setLevel(logger.INFO)


def plot_comparison_records(records, outdir, title="response_values", y_scale="log"):
    df = pd.DataFrame.from_records(records)
    ax = sns.stripplot(
        x="which_model",
        y="response_value",
        data=df,
        order=["old_target", "new_target", "old_non_target", "new_non_target"],
    )
    sns.boxplot(
        showmeans=True,
        meanline=True,
        meanprops={"color": "k", "ls": "-", "lw": 2},
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
        order=["old_target", "new_target", "old_non_target", "new_non_target"],
    )

    ax.set(yscale=y_scale)
    plt.savefig(outdir / f"{title}.stripplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    ax = sns.boxplot(
        x="which_model",
        y="response_value",
        data=df,
        order=["old_target", "new_target", "old_non_target", "new_non_target"],
    )
    ax.set(yscale=y_scale)
    plt.savefig(outdir / f"{title}.boxplot.png", dpi=150, bbox_inches="tight")


def plot_replay_comparison(new_target_eeg_evidence: np.ndarray,
                           new_non_target_eeg_evidence: np.ndarray,
                           old_target_eeg_evidence: np.ndarray,
                           old_non_target_eeg_evidence: np.ndarray,
                           outdir: str,
                           parameters: Parameters,
                           ) -> None:

    def convert_to_records(arr,
                           key_value,
                           key_name="which_model",
                           value_name="response_value") -> List[dict]:
        return [{key_name: key_value, value_name: val} for val in arr]

    records = []

    records.extend(convert_to_records(new_target_eeg_evidence, "new_target"))
    records.extend(convert_to_records(new_non_target_eeg_evidence, "new_non_target"))
    records.extend(convert_to_records(old_target_eeg_evidence, "old_target"))
    records.extend(convert_to_records(old_non_target_eeg_evidence, "old_non_target"))

    plot_comparison_records(records, outdir, y_scale="log")
