
"""evaluator.py """

from bcipy.signal.evaluate.rules import HighVoltage, LowVoltage


class Evaluator:

    """Evaluator. Takes in raw data and tests them against given
    rules, which elicit the rejection of a sequence when broken.
    Feeds warnings to artifact rejector as suggestions.

    Add rules given in parameters to evaluator's ruleset and set
    keys for broken_rules. One heading per rule.

    params (dict): dictionary of parameters from json file, given by caller
    rules (list of rule objects, defined in rules.py)

    """

    def __init__(self, params, highvoltage, lowvoltage):

        self.rules = []

        # if highvoltage threshold is enabled, add to rules
        if highvoltage:
            self.rules.append(HighVoltage('highvoltage_threshold',
                                          params["highvoltage_value"]))

        # if lowvoltage threshold is enabled, add to rules
        if lowvoltage:
            self.rules.append(LowVoltage('lowvoltage_threshold',
                                         params["lowvoltage_value"]))

    def evaluate(self, data):
        """Evaluates sequence data using selected rules
        from parameters file."""

        for rule in self.rules:

            if rule.is_broken(data):

                return False
