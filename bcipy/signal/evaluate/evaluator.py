from bcipy.signal.evaluate.rules import HighVoltage, LowVoltage


class Evaluator:
    """Evaluator. 

    Takes in raw data and tests them against given
    rules, which elicit the rejection of a sequence when broken.
    Feeds warnings to artifact rejector as suggestions.

    Add rules given in parameters to evaluator's ruleset and set
    keys for broken_rules. One heading per rule.

    parameters (dict): dictionary of parameters from json file, given by caller
    rules (list of rule objects, defined in rules.py)
    """

    def __init__(self, parameters, high_voltage, low_voltage):
        self.rules = []

        # if high_voltage threshold is enabled, add to rules
        if high_voltage:
            self.rules.append(HighVoltage(parameters['high_voltage_value']))

        # if low_voltage threshold is enabled, add to rules
        if low_voltage:
            self.rules.append(LowVoltage(parameters['low_voltage_value']))

    def evaluate(self, data):
        """Evaluate.

        Evaluates sequence data using selected rules from parameters file.
        """

        for rule in self.rules:

            if rule.is_broken(data):

                return False

        return True

    def __str__(self):
        rules = [str(rule) for rule in self.rules]
        return f'Evaluator with {rules}'
