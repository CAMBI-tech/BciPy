from abc import ABC

class Action(ABC):
    """
    Abstract class for actions
    """
    @staticmethod
    def get_all_actions():
        """
        Returns a list of all actions
        """
        return Action.__subclasses__()
    label: str
    @classmethod
    def match(cls, action_str: str) -> bool:
        """
        Determines if the action matches the given string
        """
        return action_str == cls.label

    def execute(self):
        ...


class OfflineAnalysisAction(Action):
    """
    Action for offline analysis
    """
    label = 'Offline Analysis'

    def execute(self):
        raise NotImplementedError
