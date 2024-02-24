""" Wrappers that run a simulation """
import datetime
import os
import time

from bcipy.helpers.parameters import Parameters
from bcipy.simulator.helpers import artifact
from bcipy.simulator.simulator_base import Simulator


class MultiSimRunner:

    def __init__(self, simulator: Simulator, n: int, save_dir=None, iteration_sleep=0):
        self.simulator: Simulator = simulator
        self.parameters: Parameters = self.simulator.get_parameters()
        self.n: int = n
        self.iteration_sleep: int = iteration_sleep

        self.wrapper_save_dir: str = save_dir if save_dir else self.__make_default_save_path()

    def run(self):
        # making wrapper dir for all simulations
        os.makedirs(self.wrapper_save_dir)

        for i in range(self.n):

            # creating save dir for sim_i, then mutating sim_i save_directory
            sim_i_save_dir = artifact.init_save_dir(self.wrapper_save_dir, f"run{i}")
            self.simulator.save_dir = sim_i_save_dir
            artifact.configure_logger(f"{sim_i_save_dir}/logs", f"logs")

            # running simulator
            self.simulator.run()

            if self.iteration_sleep > 0:
                time.sleep(self.iteration_sleep)

            # TODO accumulate referee metrics
            self.simulator.reset()

        # TODO save averaged results to a file in root of wrapper dir

    def __make_default_save_path(self):
        output_path = "bcipy/simulator/generated"
        now_time = datetime.datetime.now().strftime("%m-%d-%H:%M")
        dir_name = f"SIM_MULTI_{self.n}_{now_time}"

        return f"{output_path}/{dir_name}"


class SingleSimRunner:

    def __init__(self, simulator: Simulator, save_dir=None):
        self.simulator: Simulator = simulator
        self.parameters: Parameters = self.simulator.get_parameters()

        self.wrapper_save_dir: str = save_dir if save_dir else self.__make_default_save_path()

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
