from enum import Enum
from bcipy.language_model import oclm_language_model
from bcipy.language_model import prelm_language_model


class LmType(Enum):
    """Enum of the registered language model types. The types are associated
    with constructors for creating the model.
    Ex.
    >>> LmType.PRELM.model()
    """
    PRELM = prelm_language_model.LangModel
    OCLM = oclm_language_model.LangModel

    # pylint: disable=unused-argument,protected-access
    def __new__(cls, *args, **kwds):
        """Autoincrements the value of each item added to the enum."""
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, model):
        self.model = model


def LangModel(lmtype: LmType, logfile: str = "log", port: int = None):
    """Creates a new Language Model given the LmType."""

    assert lmtype, "Language Model type is required"
    model = lmtype.model
    config = model.DEFAULT_CONFIG
    if port:
        config.port = port
    return model(config, logfile)
