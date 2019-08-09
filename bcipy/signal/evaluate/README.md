# Evaluate Module

Class definitions for handling signal evaluation, and rules by which the signal is to be
evaluated.

## Evaluator

Handles signal evaluation by maintaining a set of enabled rules and testing sequences against
each rule. Is boolean in nature, informing whether a rule was broken.

__Example usage:__

	`evaluator = Evaluator('bcipy/parameters/parameters.json',True,False)` 
	Evaluator is initialised with parameters file, and whether a rule is enabled

	`evaluator.evaluate(sequence) -> True`, True meaning that no rules were broken and

	`evaluator.evaluate(sequence) -> False`, False meaning that at least one rule was broken.

__Importing:__

	The `__init__.py` file already contains instructions for importing Evaluator:

		`from bcipy.signal.evaluate.evaluator import Evaluator

		__all__ = [
		    'Evaluator'
		]`

	With this, any future script that imports Evaluator can do the following:
		
		`from bcipy.signal.evaluate import Evaluator`

### Rules

A Python abstract base class (ABC) that utilises the abstract method, is_broken, which tests an
array of data based on some criteria unique to the rule (threshold, etc.).

__Example usage:__

	`high_rule = HighVoltage(75E-6)` Voltage rules, for example, are initialised with threshold values

	`high_rule.is_broken(data) -> True`, True meaning that a rule was broken

	`high_rule.is_broken(data) -> False`, False meaning that the rule was not broken

__Importing:__

	Rules are imported individually.

	Example: `from bcipy.signal.evaluate.rules import HighVoltage, LowVoltage`