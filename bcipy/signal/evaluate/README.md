# Evaluate Module

Class definitions for handling signal evaluation and rules by which the signal is to be
evaluated. The module contains classes for detecting and correcting artifacts in the data.

## Artifact Detection

Artifact detection is the process of identifying and labeling unwanted signals from the data. This is
done by comparing the signal to a set of voltage and EOG thresholds. The thresholds are set by the user
and can be adjusted to fit the needs of the user. By default, the thresholds are set to 100uV for the
voltage threshold and 50uV for the EOG threshold. The artifact detection process is done in the following
steps:

1. The signal is passed through a bandpass filter to remove unwanted frequencies.
2. The signal is then passed through a notch filter to remove the 60Hz noise.
3. The signal is then compared to the voltage and EOG thresholds.
4. If the signal exceeds the thresholds, it is labeled as an artifact. All artifacts are annotated in the
   data file with the prefix `BAD_`.

### Artifact Detection Usage

The `ArtifactDetection` class is used to detect artifacts in the data. The class takes in a `RawData` object, a `DeviceSpec` object, and a `Parameters` object. The `RawData` object contains the data to be analyzed, the `DeviceSpec` object contains the specifications of the device used to collect the data, and the `Parameters` object contains the parameters used to detect the artifacts. The `ArtifactDetection` class has a method called `detect_artifacts` that returns a list of the detected artifacts.

```python
from bcipy.signal.evaluate.artifact import ArtifactDetection

# Assuming BciPy raw data object, device spec and parameters object are already defined.

artifact_detector = ArtifactDetection(raw_data, parameters device_spec)
detected_artifacts = artifact_detector.detect_artifacts()
```

Optionally, the user can provide session_triggers on initialization of the `ArtifactDetection` object. This is useful when the user wants to determine artifacts overlapping with triggers of interest. The `session_triggers` parameter is a tuple of lists, where the tuple contains (trigger_type, trigger_timing, trigger_label).

```python
from bcipy.signal.evaluate.artifact import ArtifactDetection

# Assuming BciPy raw data object, device spec and parameters object are already defined. Additionally, session_triggers is defined.

artifact_detector = ArtifactDetection(raw_data, parameters, device_spec, session_triggers=session_triggers)
detected_artifacts = artifact_detector.detect_artifacts()
```

This can be used in conjunction with the `ArtifactDetection` semiautomatic mode to determine artifacts that overlap with triggers of interest and correct any labels before removal. To use the semiautomatic mode, the user must provide a list of triggers of interest. The `ArtifactDetection` class can be inititalized with `semi_automatic`. The `semi_automatic` parameter is a boolean that determines if the user wants to manually correct or add to the detected artifacts.

```python

from bcipy.signal.evaluate.artifact import ArtifactDetection

# Assuming BciPy raw data object, device spec and parameters object are already defined. Additionally, session_triggers is defined.

artifact_detector = ArtifactDetection(raw_data, parameters, device_spec, session_triggers=session_triggers semi_automatic=True)
detected_artifacts = artifact_detector.detect_artifacts()
```

To output the detected artifacts to a new file, the `ArtifactDetection` class can be initialized with the `save_path` parameter. The `save_path` parameter is the path to the MNE file with the detected artifacts will be written.

```python
from bcipy.signal.evaluate.artifact import ArtifactDetection

# Assuming BciPy raw data object, device spec and parameters object are already defined. Additionally, session_triggers is defined.

artifact_detector = ArtifactDetection(raw_data, parameters, device_spec, session_triggers=session_triggers, save_path='path/to/save/file')
detected_artifacts = artifact_detector.detect_artifacts()
```

Finally, if wanting to export only the triggers with timestamps for use in another software, the detected artifcats can be exported to a txt file using the `write_mne_annotations` method.

```python
from bcipy.signal.evaluate.artifact import ArtifactDetection, write_mne_annotations

# Assuming BciPy raw data object, device spec and parameters object are already defined.

artifact_detector = ArtifactDetection(raw_data, parameters, device_spec)
detected_artifacts = artifact_detector.detect_artifacts()
write_mne_annotations(
    detected_artifacts,
    'path/to/save/file',
    'artifact_annotations.txt')
```

### Artifact Correction

Artifact correction is the process of removing unwanted signals from the data. After detection is complete, the user may use the MNE epoching tool to remove the unwanted epochs and channels.

```python
from bcipy.signal.evaluate.artifact import ArtifactDetection, mne_epochs

# Assuming BciPy raw data object, device spec and parameters object are already defined. Additionally, session_triggers is defined.
artifact_detector = ArtifactDetection(raw_data, parameters, device_spec, session_triggers=session_triggers)
# After calling this an mne_data object is created in the artifact_detector object with the annotations
detected_artifacts = artifact_detector.detect_artifacts()

# Use MNE epoching tool to remove unwanted epochs and channels
mne_data = artifact_detector.mne_data
trial_length = 0.5 # seconds
# This will return the epochs object with the bad epochs removed. If no artifact detection was done, trigger_timing and trigger_labels must be provided.
epochs = mne_epochs(mne_data, trial_length, preload=True, reject_by_annotation=True)

# This will return the epochs object with the bad epochs removed. A drop log can be accessed to see which and how many epochs were removed.
```

## Fusion Accuracy

The `calculate_eeg_gaze_fusion_acc` function is used to evaluate the performance of the BCI system. The function takes in a list of EEG and gaze data, and returns the accuracy of the fusion of the two signals. The function uses the following steps to calculate the accuracy:

The data is split into train and test sets for generating accuracy metrics. Predictions are generated for both single modal and multimodal cases, where multimodal predictions are generated by using Bayesian Fusion. The accuracy of the predictions is then calculated and returned.

### Fusion Usage

The `calculate_eeg_gaze_fusion_acc` function is used to evaluate the performance of the BCI system. The function takes in a list of EEG and gaze data, and returns the accuracy of the fusion of the two signals.

```python
from bcipy.signal.evaluate.fusion import calculate_eeg_gaze_fusion_acc

# Assuming BciPy raw data objects, device specs and parameters object are already defined.

result = calculate_eeg_gaze_fusion_acc(eeg_data, gaze_data, eeg_spec, gaze_spec, symbol_set, parameters, data_folder)
```
