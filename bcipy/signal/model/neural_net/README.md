# Notes from code review 5/14


- document assumptions that affect "outer" code:
    - Model assumes random symbol ordering during presentation.


- rather than accepting np.array everywhere, try to use data classes and/or namedtuples
    - maybe: 
        ```python
        class InquiryData:
            def __init__(self, data):
                self.data = data

            @property
            def num_channel(self):
                return self.data.shape[1]
        ```

    - instead of returning dictionary of stuff from model, can use:
        ```python
        class Employee(NamedTuple):
            name: str
            id: int
        ```

- delete DummyEEGClassifier (unused)

- move RiggedClassifier to test folder

- avoid clashing names between Trainer and model_wrapper
    - update naming to match glossary (e.g. sequences -> inquiries, ): https://github.com/CAMBI-tech/BciPy/tree/1.5.1#glossary

- metrics:
    - time to train (during a user session). Need to evaluate CPU training time
    - AUC (need to be careful about interpretation)

- test on current dataset where test_set contains unseen users


- There is only support for single session training. We should consider the bulk train in more detail; We need a bulk loader and a transformer for NN.
- Move dummy models etc to a test or demo. If useful, move all to a core file.

- Standardize our data classes (EEG etc)

- [x] remove transform

- [x] remove checkpoint (except minimal for early stop)

- [ ] change "sequence" to "inquiry"