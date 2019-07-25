
"""evaluator.py """

#Imports all rules, must be updated upon addition of new rules
from bcipy.signal.evaluate.rules import HighVoltage, LowVoltage

class Evaluator:

	"""Evaluator. Takes in chunks of raw data and tests them against given
	rules, which raise flags if broken. Feeds warnings to artifact rejector
	as suggestions. 
	
	Add rules given in parameters to evaluator's ruleset and set keys for broken_rules.
	One heading per rule.

	params (dict): dictionary of parameters from json file, given by caller
	rules (list of rule objects, defined in rules.py)
	broken_rules (dict): {'rule_name_1': True/False (indicating violation), 'rule_name_2': T/F, 
						... , 'rule_name_n': T/F}

	flag (bool): toggle flagging behaviour. A rule is flagged when it is marked 'True' in broken_rules."""

	def __init__(self,params,highvoltage,lowvoltage):

		self.rules = []

		self.broken_rules = {}

		self.flag = True

		#if highvoltage threshold is enabled, add to rules
		if highvoltage:
			self.rules.append(HighVoltage('highvoltage_threshold',params["highvoltage_value"]))
			self.broken_rules.update({'highvoltage_threshold': False})

		#if lowvoltage threshold is enabled, add to rules
		if lowvoltage:
			self.rules.append(LowVoltage('lowvoltage_threshold',params["lowvoltage_value"]))
			self.broken_rules.update({'lowvoltage_threshold': False})

	def evaluate_offline(self,data):

		"""Evaluates offline sequence data using selected rules from parameters file.
		Raises flag for that rule if broken"""

		for rule in self.rules:

			#Ignore flag behaviour if evaluating data point-by-point
			if not self.flag:
				return rule.isBroken(data)

			#flags if rule was broken
			elif rule.isBroken:

				self.broken_rules[rule.name] = True
			
				#bump flag counter
				#self.broken_rules[rule.name][1] += 1

	def reset(self):

		"""Resets broken_rules to all False """

		for key in self.broken_rules:

			self.broken_rules[key] = False