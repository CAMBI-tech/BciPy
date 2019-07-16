
"""evaluate.py """


from bcipy.signal.evaluate.rules import *
from bcipy.helpers.load import load_json_parameters

class Evaluator:

	"""Evaluator. Takes in chunks of raw data and tests them against given
	rules, which raise flags if broken. Feeds warnings to artifact rejector
	as suggestions. 
	
	Add rules given in parameters to evaluator's ruleset and set keys for broken_rules.
	One heading per rule.

	rules (list of rule objects, defined in rules.py)
	broken_rules (dict): {'rule name1': True/False (indicating violation), 'rule_name2': T/F, ... , 'rule_namen': T/F}"""

	def __init__(self):

		self.rules = []

		self.broken_rules = {}

		params_file = 'bcipy/parameters/parameters.json'

		if load_json_parameters(params_file,value_cast = True)['HighVoltage Threshold']:
			self.rules.append([HighVoltage()])
			self.broken_rules.update({HighVoltage().name: False})


		if load_json_parameters(params_file,value_cast = True)['LowVoltage Threshold']:
			self.rules.append([LowVoltage()])
			self.broken_rules.update({LowVoltage().name: False})

		#self.broken_rules = {rule.name: False for rule in self.rules}

	def evaluate_signal(self,data):

		"""Evaluates raw data using selected rules from parameters file
		raises flag for that rule if broken"""

		for rule in self.rules:

			if rule.isBroken(data) and not self.broken_rules[rule.name]:

				self.broken_rules[rule.name] = True
		