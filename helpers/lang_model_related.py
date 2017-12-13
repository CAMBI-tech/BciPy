from language_model.language_model import LangModel
import os


def init_language_model(parameters):
    try:
        abs_path_fst = os.path.abspath(parameters['path_to_fst']['value'])
        lmodel = LangModel(
            abs_path_fst,
            host=str(parameters['lang_model_server_host']['value']),
            port=str(parameters['lang_model_server_port']['value']),
            logfile="lmwrap.log")
        # init LM
        lmodel.init()

    except Exception as e:
        raise e

    return lmodel
