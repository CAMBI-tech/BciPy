"""Defines classes that can write markers to LabStreamingLayer StreamOutlet."""
import logging
from typing import Any
import pylsl

log = logging.getLogger(__name__)


class MarkerWriter():
    """Abstract base class for an object that can be used to handle stimulus
    markers.
    """

    def push_marker(self, marker: Any):
        """Push the given stimulus marker for processing.

        Parameters
        ----------
        - marker : any object that can be converted to a str
        """
        raise NotImplementedError()

    def cleanup(self):
        """Performs any necessary cleanup"""
        raise NotImplementedError()


class LslMarkerWriter(MarkerWriter):
    """Writes stimulus markers to a LabStreamingLayer StreamOutlet
    using pylsl. To consume this data, the client code would need to create a
    pylsl.StreamInlet. See https://github.com/sccn/labstreaminglayer/wiki.
    """

    def __init__(self,
                 stream_name: str = "BCI_Stimulus_Markers",
                 stream_id: str = "bci_stim_markers"):
        super(LslMarkerWriter, self).__init__()
        self.stream_name = stream_name
        markers_info = pylsl.StreamInfo(stream_name, "Markers", 1, 0, 'string',
                                        stream_id)
        self.markers_outlet = pylsl.StreamOutlet(markers_info)

    def push_marker(self, marker: Any):
        """Push the given stimulus marker for processing.

        Parameters
        ----------
        - marker : any object that can be converted to a str
        """
        log.debug(f'Pushing marker {str(marker)} at {pylsl.local_clock()}')
        self.markers_outlet.push_sample([str(marker)])

    def cleanup(self):
        """Cleans up the StreamOutlet."""
        del self.markers_outlet


class NullMarkerWriter(MarkerWriter):
    """MarkerWriter which doesn't write anything.

    A NullMarkerWriter can be passed in to the calling object in scenarios
    where marker handling occurs indirectly (ex. through a trigger box). By
    using a NullMarkerWriter rather than a None value, the calling
    object does not have to do additional null checks and a separation
    of concerns is maintained regarding how triggers are written for different
    devices.

    See the Null Object Design Pattern:
    https://en.wikipedia.org/wiki/Null_object_pattern
    """

    def push_marker(self, marker: Any):
        pass

    def cleanup(self):
        pass
