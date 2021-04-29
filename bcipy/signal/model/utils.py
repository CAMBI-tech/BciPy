from .pca_rda_kde import PcaRdaKdeModel
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def load_signal_model(filename: str = None, model_class=PcaRdaKdeModel, model_kwargs={"k_folds": 10}):
    # use python's internal gui to call file explorers and get the filename

    if not filename:
        try:
            Tk().withdraw()  # we don't want a full GUI
            filename = askopenfilename()  # show dialog box and return the path

        # except, raise error
        except Exception as error:
            raise error

    # load the signal_model with pickle
    signal_model = model_class(**model_kwargs)
    with open(filename, "rb") as f:
        signal_model.load(f)

    return (signal_model, filename)
