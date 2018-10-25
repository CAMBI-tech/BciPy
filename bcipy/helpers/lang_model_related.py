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

    port = int(parameters['lang_model_server_port'])
    selected_lmtype = parameters.get("lang_model_type", "PRELM")

    lmodel = LangModel(LmType[selected_lmtype],
                       logfile="lmwrap.log", port=port)
    lmodel.init()

    return lmodel
