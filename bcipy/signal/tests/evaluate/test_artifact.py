import unittest
from mockito import when, mock, verify
from bcipy.signal.evaluate.artifact import ArtifactDetection, DefaultArtifactParameters, ArtifactType
from bcipy.helpers import convert
class TestArtifactDetection(unittest.TestCase):

    def setUp(self):
        self.raw_data = mock()
        self.mne_data = mock()
        self.parameters = {
            'down_sampling_rate': 2,
            'notch_filter_frequency': 60,
            'filter_high': 50,
            'filter_low': 0.5,
            'filter_order': 4,
        }
        self.device_spec = mock()
        self.device_spec.content_type = 'EEG'
        self.device_spec.units = 'volts'
        channel_spec = mock()
        channel_spec.name = 'Fp1'
        channel_spec.units = 'volts'
        self.device_spec.channel_specs = [channel_spec]
        self.channel_map = []
        when(convert).convert_to_mne(self.raw_data, channel_map=self.channel_map, transform=any, volts=True).thenReturn(self.mne_data)
    
    def test_artifact_detection_init(self):
        """Test the ArtifactDetection class."""
        ar = ArtifactDetection(raw_data=self.raw_data, parameters=self.parameters, device_spec=self.device_spec)

    # def test_artifact_detection_init_throws_exception_unsupported_device(self):
    #     """Test the ArtifactDetection class."""
    #     ar = ArtifactDetection(raw_data=None, parameters=None, device_spec=None)


    # def test_artifact_detection_init_throws_exception_unsupported_units(self):
    #     """Test the ArtifactDetection class."""
    #     ar = ArtifactDetection(raw_data=None, parameters=None, device_spec=None)


    # def test_artifact_detection_init_throws_exception_no_channels(self):
    #     """Test the ArtifactDetection class."""
    #     ar = ArtifactDetection(raw_data=None, parameters=None, device_spec=None)


    def test_artifact_detection_label_artifacts(self):
        """Test the ArtifactDetection class."""
        pass

    def test_artifact_detection_label_eog_artifacts(self):
        pass

    def test_artifact_detection_label_voltage_artifacts(self):
        pass

    def test_artifact_type(self):
        """Test the ArtifactType class."""
        self.assertEqual(ArtifactType.BLINK.value, 'blink')
        self.assertEqual(ArtifactType.EOG.value, 'eog')

    def test_default_artifact_parameters(self):
        """Test the DefaultArtifactParameters class."""
        self.assertEqual(DefaultArtifactParameters.EOG_THRESHOLD.value, 75e-6)
        self.assertEqual(DefaultArtifactParameters.VOlTAGE_LABEL_DURATION.value, 0.25)
        self.assertEqual(DefaultArtifactParameters.ARTIFACT_LABELLED_FILENAME.value, 'artifacts.fif')


if __name__ == '__main__':
    unittest.main()