import numpy as np

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
        """Predicts probability distribution over target boxes given a signal
           this returns a uniform distribution across all boxes right now.

        Parameters
        ----------
            signal: 2D array that looks like (n_channels, n_samples)

        Returns
        ----------
            np.ndarray: 1D array of length n_boxes with probabilities summing to 1.
        """

        #Uniform distribution over boxes
        return np.ones(self.n_boxes) / self.n_boxes
