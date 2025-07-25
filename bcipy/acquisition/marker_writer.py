"""Defines classes that can write markers to LabStreamingLayer StreamOutlet."""
import logging
from typing import Any, Optional

import pylsl

from bcipy.config import SESSION_LOG_FILENAME

log = logging.getLogger(SESSION_LOG_FILENAME)


class MarkerWriter:
    """Abstract base class for an object that can be used to handle stimulus
    markers.
    """

    def push_marker(self, marker: Any) -> None:
        """Pushes the given stimulus marker for processing.

        This method must be implemented by subclasses.

        Args:
            marker (Any): Any object that can be converted to a string, representing
                          the stimulus marker.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError()

    def cleanup(self) -> None:
        """Performs any necessary cleanup.

        This method must be implemented by subclasses.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError()


class LslMarkerWriter(MarkerWriter):
    """Writes stimulus markers to a LabStreamingLayer StreamOutlet using pylsl.

    To consume this data, client code would typically create a `pylsl.StreamInlet`.
    See https://github.com/sccn/labstreaminglayer/wiki for more information.

    Args:
        stream_name (str, optional): The name of the LSL stream. Defaults to
                                     "BCI_Stimulus_Markers".
        stream_id (str, optional): The unique ID of the LSL stream. Defaults to
                                   "bci_stim_markers".
    """

    def __init__(self,
                 stream_name: str = "BCI_Stimulus_Markers",
                 stream_id: str = "bci_stim_markers"):
        super().__init__()
        self.stream_name: str = stream_name
        markers_info = pylsl.StreamInfo(stream_name, "Markers", 1, 0, 'string',
                                        stream_id)
        self.markers_outlet: pylsl.StreamOutlet = pylsl.StreamOutlet(markers_info)
        self.first_marker_stamp: Optional[float] = None

    def push_marker(self, marker: Any) -> None:
        """Pushes the given stimulus marker to the LSL stream.

        The marker is converted to a string and sent along with a local timestamp.
        The `first_marker_stamp` is set upon the first marker push.

        Args:
            marker (Any): Any object that can be converted to a string, representing
                          the stimulus marker.
        """
        stamp = pylsl.local_clock()
        log.info(f'Pushing marker {str(marker)} at {stamp}')
        self.markers_outlet.push_sample([str(marker)], stamp)
        if not self.first_marker_stamp:
            self.first_marker_stamp = stamp

    def cleanup(self) -> None:
        """Cleans up and releases the LSL StreamOutlet."""
        del self.markers_outlet


class NullMarkerWriter(MarkerWriter):
    """MarkerWriter which doesn't write anything.

    A `NullMarkerWriter` can be passed to calling objects in scenarios where
    marker handling occurs indirectly (e.g., through a trigger box). By using
    a `NullMarkerWriter` instead of `None`, the calling object avoids additional
    null checks, maintaining a separation of concerns regarding how triggers
    are written for different devices.

    See the Null Object Design Pattern:
    https://en.wikipedia.org/wiki/Null_object_pattern
    """

    def push_marker(self, marker: Any) -> None:
        """Overrides the abstract method to do nothing."""
        pass

    def cleanup(self):
        """Overrides the abstract method to do nothing."""
        pass
