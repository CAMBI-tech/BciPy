# BciPy Data

Core data strucutres needed for BciPy operation. These include triggers, parameters, and raw data.

## Contents

- `list`: utility methods for list processing
- `parameters`: module for functionality related to system configuration via the parameters.json file
- `raw_data`: functionality for reading and writing raw signal data
- `report`: methods for generating BciPy PDF reports
- `session`: methods for managing and parsing session.json data
- `stimuli`: methods for generating stimuli and inquiries for presentation
- `symbols`: methods for working with symbols and symbol lists. Ex. Alphabet, QWERTY, etc.
- `triggers`: methods and data classes defining BciPy internal triggering

  

## Triggers

Triggers consist of a label, type and a timestamp. These are what is used to align events both in real-time and after a session for further processing or model training. During operation of a BciPy task, Triggers are generated when display or other events of interest occur. These are timestamped using a monotonic clock (See helpers/clock.py).

A new Trigger may defined as follows:
```python
from bcipy.data.triggers import Trigger, TriggerType

# label can be any utf-8 compliant string
nontarget_trigger = Trigger('nontarget_label', TriggerType.NONTARGET, 1.0111)
```

### Trigger Types

The supported internal trigger types are as follows:

```python
NONTARGET = "nontarget"
TARGET = "target"
FIXATION = "fixation"
PROMPT = "prompt"
SYSTEM = "system"
OFFSET = "offset"
EVENT = "event"
PREVIEW = "preview"
```

### TriggerHandler

You can use the BciPy TriggerHandler to read and write any Triggers!

### Writing

You will need three pieces of information to create a new handler for writing:

1. path where trigger data should be written
2. name of the trigger file without extension
3. a defined FlushFrequency. This sets how often the handler should write the data. Incrementally (FlushFrequency.EVERY) or at session end (FlushFrequency.END).

```python
from bcipy.data.triggers import TriggerHandler, FlushFrequency

path_to_trigger_save_location = '.'
trigger_file_name = 'triggers' # BciPy will add the correct extension. Currently, .txt is used.
flush = FlushFrequency.END

handler = TriggerHandler(path_to_trigger_save_location, trigger_file_name, flush)

triggers = [
    Trigger('test_trigger', TriggerType.SYSTEM, 1.0111),
    Trigger('target', TriggerType.TARGET, 2),
    Trigger('nontarget', TriggerType.NONTARGET, 3),
]

handler.add_triggers(triggers)
handler.close() # this will call write one final time on any triggers added since last flush
```


### Loading

To load a BciPy triggers.txt file, the TriggerHandler load method can be used. Because it is a staticmethod, the class does not need to be initialized before the load method is used.

```python
from bcipy.data.triggers import TriggerHandler

path_to_trigger_save_location = './triggers.txt'

triggers = TriggerHandler.load(path_to_trigger_save_location)
# it will load in data like this:
# [Trigger: label=[starting_offset] type=[offset] time=[-13149613.488788936], Trigger: label=[x] type=[prompt] time=[4.96745322458446]]
```

Alternately, you can pass offset and exclusion as keyword arguments to modify the behavior of `TriggerHandler.load`. Offset will add time to every timestamp loaded. Use this to correct any static system offsets! The default value for offset is 0.0. Exclusions can be used to pre-filter any unwanted Triggers, such as SYSTEM.

```python
from bcipy.data.triggers import TriggerHandler

path_to_trigger_save_location = './triggers.txt'

# exclude system triggers
triggers = TriggerHandler.load(path_to_trigger_save_location, exclusion=[TriggerType.SYSTEM])

# apply a 2 second offset to all timestamps loaded
triggers = TriggerHandler.load(path_to_trigger_save_location, offset=2.0)
```

### Trigger File Format

Triggers as written from BciPy tasks are assumed to have the following structure,

1. A single trigger is written per line with three columns (label type timestamp).
2. If a clock offset is present and subsequent triggers should be corrected, it may be written with the trigger type `offset`. The value of that Trigger will be added to any timestamp values written in the file. When using various clocks that don't start at zero, this can be used to align the data to similar t=0. If time should be subtracted, write a negative value as demonstrated below for both offset values. BciPy will only apply the first instance of offset, but will return all Triggers and the additional offsets may be applied as desired. In the example below, only starting_offset would be applied to all other triggers; another_offset would be returned.
3. Only a valid TriggerType may be read.

```
starting_offset offset -3421.2852307
another_offset offset -2
N prompt 3490.3607581
+ fixation 3491.3668763
Y nontarget 3491.8722132
P nontarget 3492.0780858
J nontarget 3492.2839911
F nontarget 3492.4900032
D nontarget 3492.6959695
K nontarget 3492.9021379
X nontarget 3493.1079959
< nontarget 3493.3139829
M nontarget 3493.5198677
N target 3493.7257014
X prompt 3495.4642608
+ fixation 3496.4697921
Z nontarget 3496.9749562
```

### Multiple Devices

When multiple devices are in use, each device should provide its own starting_offset trigger. The trigger time should be the timestamp associated with the first sample in the data for that device.

```python
from bcipy.data.triggers import Trigger, TriggerType, offset_label

# label can be any utf-8 compliant string
eeg_offset = Trigger(offset_label(device_type='EEG'), TriggerType.OFFSET, -3400.0)
eyetracker_offset = Trigger(offset_label(device_type='EYETRACKER'), TriggerType.OFFSET, -3450.0)
```

After adding other triggers, the resulting file will look something like:

```
starting_offset offset -3400.0
starting_offset_EYETRACKER offset -3450.0
N prompt 3490.3607581
+ fixation 3491.3668763
Y nontarget 3491.8722132
...
```

Triggers can then be loaded with timestamps relative to a device's start.

```python
from bcipy.data.triggers import trigger_decoder, TriggerType

types, times, labels = trigger_decoder('triggers.txt', device_type='EEG')
# types == ['prompt', 'fixation', 'prompt']
# times == [90.36075810000011, 91.36687630000006, 91.8722131999998]
# labels == ['N', '+', 'Y']

_, times, _ = trigger_decoder('triggers.txt', device_type='EYETRACKER')
# times == [40.36075810000011, 41.36687630000006, 41.872213199999806]
```
