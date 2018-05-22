import pylsl
import logging


class MarkerWriter(object):
    """Abstract base class for an object that can be used to handle stimulus
    markers.
    """

    def push_marker(self, marker: any, lsl_time: float=None):
        """Push the given stimulus marker for processing.

        Parameters:
            marker : any object that can be converted to a str
            lsl_time : optional float indicating the time to associate with
                the marker.
        """
        raise NotImplementedError()

    def now(self) -> float:
        """Returns current time from the clock internal to the marker
        writer."""
        raise NotImplementedError()

    def cleanup(self):
        """Performs any necessary cleanup"""
        raise NotImplementedError()


class LslMarkerWriter(object):
    """Writes stimulus markers to a LabStreamingLayer StreamOutlet
    using pylsl. To consume this data, the client code would need to create a
    pylsl.StreamInlet. See https://github.com/sccn/labstreaminglayer/wiki.
    """

    def __init__(self):
        super(LslMarkerWriter, self).__init__()
        markers_info = pylsl.StreamInfo("BCI Stimulus Markers",
                                        "Markers", 1, 0, 'string',
                                        "bci_stim_markers")
        self.markers_outlet = pylsl.StreamOutlet(markers_info)
        self._stamp = None

    def now(self) -> float:
        """Create the timestamp for the next marker using the pylsl
        local_clock."""
        return pylsl.local_clock()

    def push_marker(self, marker: any, lsl_time: float=None):
        """Push the given stimulus marker for processing.
        Parameters:
            marker : any object that can be converted to a str
            lsl_time : optional float indicating the time to associate with
                the marker. If omitted, the pylsl.local_clock() will be used.
        """
        stamp = lsl_time if lsl_time is not None else self.now()
        self.markers_outlet.push_sample([str(marker)], stamp)
        logging.debug(f"Pushing marker: {marker}; timestamp: {stamp}")

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

    def push_marker(self, marker: any, lsl_time: float=None):
        pass

    def now(self) -> float:
        return 0.0

    def cleanup(self):
        pass
