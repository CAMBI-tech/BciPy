import unittest
from bcipy.signal.evaluate.artifact import ArtifactDetection, DefaultArtifactParameters, ArtifactType

class TestArtifactDetection(unittest.TestCase):

    def setUp(self):
        self.raw_data = None
        self.parameters = None
        self.device_spec = None
    
    def test_artifact_detection_init(self):
        """Test the ArtifactDetection class."""
        ar = ArtifactDetection(raw_data=None, parameters=None, device_spec=None)

    def test_artifact_detection_init_throws_exception_unsupported_device(self):
        """Test the ArtifactDetection class."""
        ar = ArtifactDetection(raw_data=None, parameters=None, device_spec=None)


    def test_artifact_detection_init_throws_exception_unsupported_units(self):
        """Test the ArtifactDetection class."""
        ar = ArtifactDetection(raw_data=None, parameters=None, device_spec=None)


    def test_artifact_detection_init_throws_exception_no_channels(self):
        """Test the ArtifactDetection class."""
        ar = ArtifactDetection(raw_data=None, parameters=None, device_spec=None)


    def test_artifact_detection_label_artifacts(self):
        """Test the ArtifactDetection class."""
        pass

    def test_artifact_detection_label_eog_artifacts(self):
        pass

    def test_artifact_detection_label_voltage_artifacts(self):
        pass

    def test_artifact_type(self):
        """Test the ArtifactType class."""
        self.assertEqual(ArtifactType.BLINK, 'blink')
        self.assertEqual(ArtifactType.EOG, 'eye_movement')

    def test_default_artifact_parameters(self):
        """Test the DefaultArtifactParameters class."""
        self.assertEqual(DefaultArtifactParameters.EOG_THRESHOLD.value, 0.5)
        self.assertEqual(DefaultArtifactParameters.VOlTAGE_LABEL_DURATION.value, 0.5)
        self.assertEqual(DefaultArtifactParameters.ARTIFACT_LABELLED_FILENAME, 'artifacts.fif')


if __name__ == '__main__':
    unittest.main()