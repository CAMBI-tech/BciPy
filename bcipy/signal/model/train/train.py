from abc import ABC

class ModelTrain(ABC):

    def __init__(self, parameters):
        
        self.parameters = parameters

    def offline(self, data_folder: str = None, alert_finished: bool = True):
        ...

    def online(self, data_folder: str = None, alert_finished: bool = True):
        ...