import unittest
from mockito import when, mock, unstub
from bcipy.signal.evaluate.artifact import ArtifactDetection, DefaultArtifactParameters, ArtifactType
from bcipy.signal.evaluate import artifact


class TestArtifactDetection(unittest.TestCase):

    def setUp(self):
        self.raw_data = mock()
        self.raw_data.total_seconds = 10
        self.raw_data.sample_rate = 1000
        self.raw_data.channels = ['Fp1', 'Fp2']
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
        self.channel_spec = mock()
        self.channel_spec.name = 'Fp1'
        self.channel_spec.units = 'volts'
        self.device_spec.channel_specs = [self.channel_spec]
        self.device_spec.analysis_channels = ['Fp1']
        self.channel_map = [1, 0]
        when(artifact).convert_to_mne(
            self.raw_data,
            channel_map=self.channel_map,
            transform=any,
            volts=True).thenReturn(
            self.mne_data)
        
    def tearDown(self) -> None:
        unstub()

    def test_artifact_detection_init(self):
        """Test the ArtifactDetection class."""
        ar = ArtifactDetection(raw_data=self.raw_data, parameters=self.parameters, device_spec=self.device_spec)
        self.assertIsInstance(ar, ArtifactDetection)
        self.assertFalse(ar.analysis_done)
        self.assertIsNone(ar.dropped)
        self.assertIsNone(ar.eog_annotations)
        self.assertIsNone(ar.voltage_annotations)

    def test_artifact_detection_init_throws_exception_unsupported_device(self):
        """Test the ArtifactDetection class throws exception when using a device that is not supported."""
        self.device_spec.content_type = 'MEG'
        with self.assertRaises(AssertionError):
            ArtifactDetection(
                raw_data=self.raw_data,
                parameters=self.parameters,
                device_spec=self.device_spec)

    def test_artifact_detection_init_throws_exception_unsupported_units(self):
        """Test the ArtifactDetection class throws exception with unsupported units."""
        self.device_spec.units = 'unsupported'
        self.channel_spec.units = 'unsupported'

        with self.assertRaises(AssertionError):
            ArtifactDetection(
                raw_data=self.raw_data,
                parameters=self.parameters,
                device_spec=self.device_spec)

    def test_artifact_detection_init_throws_exception_no_channels(self):
        """Test the ArtifactDetection class throws an exception if no channels are provided."""
        self.device_spec.channel_specs = []

        with self.assertRaises(AssertionError):
            ArtifactDetection(
                raw_data=self.raw_data,
                parameters=self.parameters,
                device_spec=self.device_spec)

    def test_artifact_detection_label_artifacts(self):
        """Test the ArtifactDetection class label artifacts method."""
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
