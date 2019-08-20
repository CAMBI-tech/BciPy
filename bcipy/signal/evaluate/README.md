# Evaluate Module

Class definitions for handling signal evaluation and rules by which the signal is to be
evaluated.

## Evaluator

Handles signal evaluation by maintaining a set of enabled rules and testing sequences against
each rule. Is boolean in nature, informing whether a rule was broken.

### Example usage

Evaluator is initialized with parameters file and whether a rule is enabled. In this case, high and low voltage rules are configured.

```python
# configuration
parameter_location = 'bcipy/parameters/parameters.json'
high_voltage_rule = True
low_voltage_rule = False

# init evaluator
evaluator = Evaluator(parameter_location, high_voltage_rule, low_voltage_rule)

# Evaluate if sequence is good (returns True / False)
evaluator.evaluate(sequence)
```

### Importing

The `__init__.py` file already contains instructions for importing Evaluator:

```python
from bcipy.signal.evaluate.evaluator import Evaluator

__all__ = [
	'Evaluator'
]
```

With this, any future script that imports Evaluator can do the following:
	
`from bcipy.signal.evaluate import Evaluator`

## Rules

A Python abstract base class (ABC) that utilizes the abstract method, is_broken, which tests an
array of data based on some criteria unique to the rule (threshold, etc.).

### Example usage

Voltage rules, for example, are initialized with threshold values

`high_rule = HighVoltage(75E-6)`


Determine if rule is broken (returns True / False)

`high_rule.is_broken(data)`

### Importing

`from bcipy.signal.evaluate.rules import HighVoltage, LowVoltage`