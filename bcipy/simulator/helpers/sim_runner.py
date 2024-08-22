""" Wrappers that run a simulation """
import dataclasses
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, NamedTuple

from bcipy.simulator.helpers import artifact
from bcipy.simulator.helpers.metrics import SimMetrics, average_sim_metrics
from bcipy.simulator.simulator_base import Simulator

log = logging.getLogger(artifact.TOP_LEVEL_LOGGER_NAME)


class RunSummary(NamedTuple):
    """Summary for a simulation run."""
    run: int
    target_text: str
    typed_text: str
    metrics: SimMetrics

    def to_dict(self) -> Dict:
        """Convert to a dictionary"""
        d = self._asdict()  # pylint: disable=no-member
        d['metrics'] = dataclasses.asdict(self.metrics)
        return d


class SimRunner():
    """Runs one or more iterations of the simulation."""

    def __init__(self,
                 save_dir: str,
                 simulator: Simulator,
                 runs: int = 1,
                 sim_log: str = artifact.DEFAULT_LOGFILE_NAME,
                 iteration_sleep: int = 0):
        self.simulator = simulator
        self.runs = runs
        self.iteration_sleep = iteration_sleep
        self.save_dir = save_dir
        self.sim_log = sim_log

    def setup_complete(self) -> bool:
        """Check if setup has already been run."""
        return Path(self.save_dir).is_dir() and self.simulator is not None

    def run(self):
        """Run one or more simulations"""
        assert self.setup_complete(), "Setup is not complete."
        self.simulator.parameters.save(self.save_dir)
        log.info("Starting simulation...\n")

        run_summaries = []
        # running simulator n times. resetting after each run
        for i in range(self.runs):
            summary = self.do_run(i)
            log.info(summary.to_dict())
            run_summaries.append(summary)
            time.sleep(self.iteration_sleep)

        self.summarize_metrics(run_summaries)
        log.info(f"\nResults logged to {self.save_dir}/{self.sim_log}")

    def do_run(self, run: int) -> RunSummary:
        """Do a simulator run and return the computed summary."""

        path = self.configure_run_directory(run)
        self.simulator.save_dir = path
        self.simulator.run()

        state = self.simulator.state_manager.get_state()
        run_metrics = self.simulator.referee.score(self.simulator)

        summary = RunSummary(run + 1,
                             target_text=state.target_sentence,
                             typed_text=state.current_sentence,
                             metrics=run_metrics)
        self.simulator.reset()
        return summary

    def configure_run_directory(self, run: int) -> str:
        """Create the necessary directories and configure the logger.
        Returns the run directory.
        """
        run_name = f"run_{run + 1}"
        path = f"{self.save_dir}/{run_name}"
        os.mkdir(path)
        artifact.configure_logger(log_path=path,
                                  file_name=f"{run_name}.log",
                                  use_stdout=False)
        return path

    def summarize_metrics(self, run_summaries: List[RunSummary]) -> None:
        """Summarize metrics from all runs."""
        all_run_metrics = [run.metrics for run in run_summaries]
        self.__save_average_metrics(average_sim_metrics(all_run_metrics))

        summary_dict = {
            summary.run: summary.to_dict()
            for summary in run_summaries
        }
        self.save_run_summaries(summary_dict)

    def __save_average_metrics(self, avg_metric_dict: Dict):
        """ Write average dictionary of metrics to a json file in root of save dir."""

        with open(f"{self.save_dir}/avg_metrics.json", 'w',
                  encoding='utf8') as output_file:
            json.dump(avg_metric_dict, output_file, indent=2)

    def save_run_summaries(self, summary_dict: Dict):
        """Write the summary for each run in the save directory."""
        with open(f"{self.save_dir}/run_summaries.json", 'w',
                  encoding='utf8') as output:
            json.dump(summary_dict, output, indent=2)
