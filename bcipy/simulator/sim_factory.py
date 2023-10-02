from pathlib import Path

from bcipy.config import DEFAULT_PARAMETER_FILENAME
from bcipy.simulator.sim_copy_phrase import SimulatorCopyPhraseReplay
from bcipy.simulator.simulator_base import Simulator


class SimulationFactory:

    @staticmethod
    def create(
            sim_task="",
            parameter_path="",
            smodel_files=None,
            lmodel_files=None,
            data_folders=None,
            out_dir="",
            **kwargs) -> Simulator:
        if sim_task == 'RSVP_COPY_PHRASE':
            # TODO validate arguments

            if not parameter_path:
                data_folder = data_folders[0]
                parameter_path = Path(data_folder, DEFAULT_PARAMETER_FILENAME)

            return SimulatorCopyPhraseReplay(parameter_path, out_dir, smodel_files, lmodel_files, data_folders[0],
                                             verbose=kwargs.get('verbose', False))

        # TODO refactor for sampling simulator
