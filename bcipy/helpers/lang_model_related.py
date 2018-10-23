import os
from bcipy.language_model.lm_modes import LangModel, LmType


def init_language_model(parameters):
    """
    Init Language Model.

    Function to Initialize remote language model and get an instance of
        LangModel wrapper. Assumes a docker image is already loaded.

    See language_model/demo/ for more information of how it works.

    Parameters
    ----------
        parameters : dict
            configuration details and path locations

    Returns
    -------
        lmodel: instance
            instance of lmodel wrapper with connections to docker server
    """

    # TODO: consider parameterizing
        # host=str(parameters['lang_model_server_host']),
        # port=str(parameters['lang_model_server_port']),
    # TODO: select language model type from params.
    lmodel = LangModel(LmType.PRELM, logfile="lmwrap.log")
    lmodel.init()

    return lmodel
