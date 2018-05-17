import pylsl
import logging

class MarkerWriter(object):
    """Abstract base class for an object that can be used to handle stimulus
    markers.
    """

    def push_marker(self, marker, lsl_time=None):
        pass

    def now(self):
        return 0.0

    def cleanup(self):
        pass


class LslMarkerWriter(object):
    """Writes stimulus markers to an LSL StreamOutlet."""

    def __init__(self):
        super(LslMarkerWriter, self).__init__()
        markers_info = pylsl.StreamInfo("BCI Stimulus Markers",
                                        "Markers", 1, 0, 'string',
                                        "bci_stim_markers")
        self.markers_outlet = pylsl.StreamOutlet(markers_info)
        self._stamp = None

    def now(self):
        """Create the timestamp for the next marker."""
        return pylsl.local_clock()

    def push_marker(self, marker, lsl_time=None):
        stamp = lsl_time if lsl_time is not None else self.now()
        self.markers_outlet.push_sample([str(marker)], stamp)
        logging.debug(f"Pushing marker: {marker}; timestamp: {stamp}")

    def cleanup(self):
        del self.markers_outlet


class NullMarkerWriter(MarkerWriter):
    """MarkerWriter which doesn't write anything."""

    def push_marker(self, marker, lsl_time=None):
        pass

    def now(self):
        return 0.0

    def cleanup(self):
        pass
