import numpy as np
from bcipy.config import DEFAULT_DEVICE_SPEC_FILENAME
import bcipy.acquisition.devices as devices
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
#added for device spec using path
from pathlib import Path

class VEPSignalModel:
    """ Loads in VEP template and returns a probability distribution over the target boxes

    Parameters
    ----------
        template_file: csv file with VEP templates
        template: An array, (n_boxes, n_channels, n_samples), that holds the templates
        n_boxes: Number of boxes used in the templates loaded
        n_channels: Number of channels for each template
        n_samples: Number of samples for each channel
    """

    def __init__(self, template_file: str):
        """
        Initialize the model and load templates from a CSV file.

        Parameters
        ----------
            template_file: csv file with VEP templates
        """
        self.template_file = template_file
        self.template = None
        self.n_boxes = 0
        self.n_channels = 0
        self.n_samples = 0
        self.load(template_file)

    def load(self, template_file: str):
        """Load VEP templates from a CSV file.

        Each line of the file should have the format (with a blank line seperating the boxes):
        '
        01_box2: -132.9836,705.1920,-587.3110, ...

        01_box2: -132.9836,705.1920,-587.3110, ...
        '

        The templates are stored in a numpy array:
        (n_boxes, n_channels, n_samples).

        Parameters
        ----------
            template_file: csv file with VEP templates
        """

        template_path = "/".join(template_file.split("/")[:-1])
        devices_by_name = devices.load(
            Path(template_path, DEFAULT_DEVICE_SPEC_FILENAME), replace=True)

        self.device_spec = devices_by_name.get("openbci_eeg")

        with open(template_file, 'r') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        #Parse each line into a list
        raw = []
        for line in lines:
            try:
                #Split at the first ':' to separate label/data
                _, data_str = line.split(':', 1)
            except ValueError:
                raise ValueError(f"Template does not match format on '{line}'")

            #Convert comma-separated template values
            values = [float(tok) for tok in data_str.strip().split(',') if tok]
            raw.append(values)

        #Convert to numpy array that looks like (n_rows, n_samples)
        array = np.array(raw)
        n_rows, n_samples = array.shape

        #3 channels always per box (O1, Oz, O2)
        n_channels = 3
        if n_rows % n_channels != 0:
            raise ValueError(
                f"Number of template lines is not a multiple of channels"
            )

        n_boxes = n_rows // n_channels
        #Reshape into (n_boxes, n_channels, n_samples)
        #lines are expected grouped by box, then channel to follow order
        self.template = array.reshape(n_boxes, n_channels, n_samples)
        self.n_boxes = n_boxes
        self.n_channels = n_channels
        self.n_samples = n_samples

    def predict(self, signal: np.ndarray) -> np.ndarray:
        """Predicts probability distribution over target boxes given a signal.

        For each of the three trials of each box, (O1, Oz, O2) run a CCA between
        the vep template and signal recieved by calculating the correlations.
        After, take the average of all the channels for each boc and normalize
        there score to sum to one.

        Parameters
        ----------
            signal: 2D array that looks like (n_channels, n_samples)

        Returns
        ----------
            np.ndarray: 1D array of length n_boxes with average correlations summing to 1.
        """

        #Holds average correlation for each box
        average_corr = []
        testing_scaler = StandardScaler()
        cca = CCA(n_components=1)

        for box in range(self.n_boxes):
            channel_correlations = []
            for channel in range(self.n_channels):
                template_signal = self.template[box, channel]
                target_signal  = signal[channel]

                #min_length = min(template_eeg.shape[0], testing_eeg.shape[0])
                min_length = min(template_signal.shape[0], target_signal.shape[0])
                #Shorten template/target to the first min_length sample and reshape according
                template_window = template_signal[:min_length].reshape(-1, 1)
                target_window = target_signal[:min_length].reshape(-1, 1)

                #Normalized template data by shifting so that average value is exactly zero
                #testing_eeg = testing_scaler.fit_transform(testing_eeg)
                template_scaled = testing_scaler.fit_transform(template_window)
                target_scaled = testing_scaler.fit_transform(target_window)

                # A, B = cca.fit_transform(template_eeg, testing_eeg)
                A, B = cca.fit_transform(template_scaled, target_scaled)
                # corri = np.corrcoef(A, B, rowvar = False)[0, 1])
                corr = np.corrcoef(A, B, rowvar=False)[0, 1]
                channel_correlations.append(corr)

            #After processing, compute the mean correlation for this box
            average_corr.append(np.mean(channel_correlations))
        #Convert list of box average_corrs to numpy array
        average_corr = np.array(average_corr)
        #Total fo rsum of all box scores (normalization)
        total = np.sum(average_corr)
        #Non-positive result like if correlations all equal 0 or negative
        if total <= 0:
            return np.ones(self.n_boxes) / self.n_boxes
        #Box average_corr summed to one and returns an array of probabilities
        return average_corr / total
