from bcipy.language_model.language_model import LangModel
import os


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

    try:
        # get the absolute path for fst from parameters
        abs_path_fst = os.path.abspath(parameters['path_to_fst'])

        # Try getting an instance of LM
        lmodel = LangModel(
            abs_path_fst,
            host=str(parameters['lang_model_server_host']),
            port=str(parameters['lang_model_server_port']),
            logfile="lmwrap.log")
        # init LM
        lmodel.init()

    except Exception as e:
        raise e

    return lmodel
