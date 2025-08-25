# BciPy Helpers

Modules necessary for BciPy system operation. These range from system utilities and triggering to core module integrations/configuration.

## Contents

- `acquisition`: helper methods for working with the acquisition module. Initializes EEG acquisition given parameters
- `clock`: clocks that provides high resolution timestamps
- `copy_phrase_wrapper`: basic copy phrase task duty cycle wrapper. Coordinates activities around evidence management, decision-making, generation of inquiries, etc
- `language_model`: helper methods for working with the language module
- `utils`: utilities for extracting git version, system information and handling of logging
- `task`: common task methods and utilities, including Trial and InquiryReshaper.
- `validate`: methods for validating experiments and fields
- `visualization`: method for visualizing EEG data collected in BciPy. Used in offline_analysis.py.
