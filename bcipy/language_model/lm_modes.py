from enum import Enum
import os
from os.path import dirname, realpath
from pathlib import Path
from typing import Dict, List
from bcipy.language_model.lm_server import LmServerConfig
from bcipy.language_model import oclm_language_model
from bcipy.language_model import prelm_language_model
from bcipy.helpers.system_utils import dot


class lmtype:
    def __init__(self, lmtype):

        self.type = lmtype
        self.host = "127.0.0.1"
        self.dport = "5000"

        if lmtype == 'oclm':
            self.port = "6000"
            self.image = "oclmimage:version2.0"
            self.nbest = "1"

        elif lmtype == 'prelm':
            self.port = "5000"
            self.image = "lmimage:version2.0"
            self.localfst = str(Path(os.path.dirname(os.path.realpath(
                __file__))) / "fst" / "brown_closure.n5.kn.fst")


class LmType(Enum):
    """Enum of the registered language model types.
    """
    PRELM = 1
    OCLM = 2

# Docker configs for each type.
LmServerConfigs = {
    LmType.PRELM: LmServerConfig(
        image="lmimage:version2.0",
        port=5000,
        docker_port=5000,
        volumes={dot(__file__, 'fst', 'brown_closure.n5.kn.fst'):
                 "/opt/lm/brown_closure.n5.kn.fst"}),
    LmType.OCLM: LmServerConfig(
        image="oclmimage:version2.0",
        port=6000,
        docker_port=5000)}

LmModels = {LmType.PRELM: prelm_language_model.LangModel,
            LmType.OCLM: oclm_language_model.LangModel}


def LangModel(lmtype: LmType, logfile: str = "log", port: int = None):
    """Creates a new Language Model given the LmType."""
    config = LmServerConfigs[lmtype]
    if port:
        config.port = port
    return LmModels[lmtype](config, logfile)
