# Simulator Demo

This module demonstrates how the simulator can be extended for various use cases.


## Multimodal

The `switch_data_processor` and `switch_model` are used to demonstrate a multimodal simulations using a button/switch as an example. To run a simulation with these inputs, you will need to perform the following steps:

1. Ensure that the switch signal model can be loaded or create a switch signal model. To create a new one:

    ```
    from pathlib import Path
    from bcipy.acquisition.datastream.mock.switch import switch_device
    from bcipy.io.save import save_model
    from bcipy.simulator.demo.switch_model import SwitchModel
    from bcipy.signal.model.base_model import SignalModel, SignalModelMetadata

    dirname = "" # TODO: enter the directory
    model = SwitchModel()
    model.metadata = SignalModelMetadata(device_spec=switch_device(), evidence_type="BTN", transform=None)
    save_model(model, Path(dirname, "switch_model.pkl"))
    ```

2. Ensure that the devices.json file has an entry for a switch

    ```
    {
        "name": "Switch",
        "content_type": "MARKERS",
        "channels": [
            { "name": "Marker", "label": "Marker" }
        ],
        "sample_rate": 0.0,
        "description": "Switch used for button press inputs",
        "excluded_from_analysis": [],
        "status": "active",
        "static_offset": 0.0
    }
    ```

3. Set the appropriate simulation parameters in the parameters.json file.

    - set the `acq_mode` parameter to 'EEG+MARKERS'.
    - ensure that `preview_inquiry_progress_method` parameter is set to "1" or "2".
    - You may also want to set the `summarize_session` parameter to `true` to see how the evidences get combined during decision-making.

4. Run a simulation.

    - Select both the EEG and the Button models
    - Use the InquirySampler

Run with verbose mode and inspect the detailed run logs to ensure that the evidence is being sampled correctly.

### Expected Behavior

Along with the 'eeg' evidence, the output session.json (and session.xlsx) should record 'btn' evidence for each inquiry. These evidences should be fused
to provide the 'likelihood' values.

For inquiries in which the target is shown:

- evidence values for symbols in the inquiry should be boosted relative to non-inquiry symbols (default values are 0.95 for boosted and 0.05 for degraded).

For inquiries in which the target not shown:

- evidence values for symbols in inquiry should be degraded

Note that the progress method (`preview_inquiry_progress_method` parameter) doesn't matter if it is set to "press to accept" or "press to skip", since the SwitchDataProcessor interprets this and outputs a 1.0 for inquiries that should be supported and 0.0 for those that shouldn't.

### Limitations

- A `preview_inquiry_progress_method` of 0 is currently not supported and an exception will be thrown. Ideally, all inquiries should get an evidence value of 1.0 (no change) with this mode.
- Button evidence only works correctly with the InquirySampler. This is due to all trials in the same inquiry receiving the same value.
