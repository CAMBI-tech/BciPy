from bcipy.signal.model import PcaRdaKdeModel
from bcipy.signal.model.base_model import SignalModelMetadata
from bcipy.helpers.save import save_model
from pathlib import Path
import numpy as np
from typing import List
from bcipy.acquisition.devices import DeviceSpec
from bcipy.signal.process import Composition


def train_bcipy_model(
        data: np.ndarray,
        labels: List[int],
        output_path: str,
        device_spec: DeviceSpec,
        default_transform: Composition) -> PcaRdaKdeModel:
    """
    Train a BciPy model on the data and labels provided.

    Note: data is (epochs, channels, samples) but the model expects (channels, epochs, samples).
    """
    # reorder data to (channels, epochs, samples)
    data = data.transpose(1, 0, 2)
    model = PcaRdaKdeModel()
    model.fit(data, labels)
    model.metadata = SignalModelMetadata(device_spec=device_spec,
                                         transform=default_transform,
                                         evidence_type="ERP",
                                         auc=model.auc)
    print(f"Training complete [AUC={model.auc:0.4f}].")
    save_model(model, Path(output_path, f"of_model_{model.auc:0.4f}.pkl"))
    return model
