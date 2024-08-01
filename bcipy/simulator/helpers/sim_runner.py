""" Wrappers that run a simulation """
import datetime
import json
import os
import time
from typing import Dict

from bcipy.helpers.parameters import Parameters
from bcipy.simulator.helpers import artifact
from bcipy.simulator.helpers.metrics import average_sim_metrics
from bcipy.simulator.simulator_base import Simulator


class SimRunner:
    """An object that orchestrates one or more simulation runs."""

    def run(self):
        """Run the simulation"""
        ...


class MultiSimRunner(SimRunner):
    """Runs multiple iterations of the simulation."""

    def __init__(self,
                 simulator: Simulator,
                 n: int,
                 save_dir=None,
                 iteration_sleep=0):
        self.simulator: Simulator = simulator
        self.parameters: Parameters = self.simulator.get_parameters()
        self.n: int = n
        self.iteration_sleep: int = iteration_sleep

        self.wrapper_save_dir: str = save_dir if save_dir else self.__make_default_save_path(
        )

    def run(self):
        # making wrapper dir for all simulations
        os.makedirs(self.wrapper_save_dir)

        # list of metrics from sim runs
        all_run_metrics = []

        # running simulator n times. resetting after each run
        for i in range(self.n):

            # creating save dir for sim_i, then mutating sim_i save_directory
            sim_i_save_dir = artifact.init_save_dir(self.wrapper_save_dir,
                                                    f"run{i}")
            self.simulator.save_dir = sim_i_save_dir
            artifact.configure_logger(f"{sim_i_save_dir}/logs", "logs")

            # running simulator
            self.simulator.run()

            if self.iteration_sleep > 0:
                time.sleep(self.iteration_sleep)

            # aggregating metrics from each run
            run_metrics = self.simulator.referee.score(self.simulator)
            all_run_metrics.append(run_metrics)

            self.simulator.reset()

        # save averaged results to a file in root of wrapper dir
        average_metrics = average_sim_metrics(all_run_metrics)
        self.__save_average_metrics(average_metrics)

    def __make_default_save_path(self):
        output_path = "bcipy/simulator/generated"
        now_time = datetime.datetime.now().strftime("%m-%d-%H:%M")
        dir_name = f"SIM_MULTI_{self.n}_{now_time}"

        return f"{output_path}/{dir_name}"

    def __save_average_metrics(self, avg_metric_dict: Dict):
        """ Writing average dictionary of metrics to a json file in root of MultiRun save dir """

        with open(f"{self.wrapper_save_dir}/avg_metrics.json",
                  'w') as output_file:
            json.dump(avg_metric_dict, output_file, indent=1)


class SingleSimRunner(SimRunner):
    """Runs the simulation once."""

    def __init__(self, simulator: Simulator, save_dir=None):
        self.simulator: Simulator = simulator
        self.parameters: Parameters = self.simulator.get_parameters()

        self.wrapper_save_dir: str = save_dir if save_dir else self.__make_default_save_path(
        )

    def run(self):
        # logging and save details
        now_time = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_dir_path = artifact.init_save_dir(self.wrapper_save_dir, now_time)

        # configuring logger to save to {output_wrapper_path}/logs
        log_path = f"{save_dir_path}/logs"
        artifact.configure_logger(log_path, now_time)

        # mutating simulator to save to specific save_dir
        self.simulator.save_dir = save_dir_path

        # running sim
        self.simulator.run()

    def __make_default_save_path(self):
        output_path = "bcipy/simulator/generated"
        return f"{output_path}"
