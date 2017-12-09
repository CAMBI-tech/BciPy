from language_model.language_model import LangModel


def init_language_model(parameters):
    lmodel = LangModel(
        parameters['path_to_fst']['value'],
        host=parameters['lang_model_server_host']['value'],
        port=parameters['lang_model_server_port']['value'],
        logfile="lmwrap.log")
    # init LM
    lmodel.init()

    return lmodel
