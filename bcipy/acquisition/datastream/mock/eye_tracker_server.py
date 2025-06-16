"""Mock eye tracker that streams data using LabStreamingLayer"""

import logging
import math
import time
from typing import Generator, List

from numpy.random import uniform

from bcipy.acquisition.datastream.lsl_server import LslDataServer
from bcipy.acquisition.devices import DeviceSpec
from bcipy.config import SESSION_LOG_FILENAME

log = logging.getLogger(SESSION_LOG_FILENAME)


def eye_tracker_device() -> DeviceSpec:
    """Mock DeviceSpec for an eye tracker.

    Returns:
        DeviceSpec: A DeviceSpec object configured for an eye tracker.
    """
    return DeviceSpec(name='EyeTracker',
                      channels=[
                          'leftEyeX', 'leftEyeY', 'rightEyeX', 'rightEyeY',
                          'leftPupilArea', 'rightPupilArea',
                          'pixelsPerDegreeX', 'pixelsPerDegreeY'
                      ],
                      sample_rate=500,
                      content_type='Gaze')


def eye_tracker_data_generator(display_x: int = 1920, display_y: int = 1080) -> Generator[List[float], None, None]:
    """Generates sample eye tracker data.

    TODO: determine appropriate values for pixelsPerDegree fields.
    TODO: look info alternatives; maybe PyGaze.
    http://www.pygaze.org/about/

    Args:
        display_x (int): The width of the display in pixels. Defaults to 1920.
        display_y (int): The height of the display in pixels. Defaults to 1080.

    Yields:
        List[float]: A list of float values representing eye tracker data, including
                     left eye X/Y, right eye X/Y, left pupil area, right pupil area,
                     pixels per degree X, and pixels per degree Y.
    """

    def area(diameter: float) -> float:
        return math.pi * (diameter / 2.0)**2

    while True:
        yield [
            float(int(uniform(0, display_x))),  # left eye x
            float(int(uniform(0, display_y))),  # left eye y
            float(int(uniform(0, display_x))),  # right eye x
            float(int(uniform(0, display_y))),  # right eye y
            area(uniform(2, 8)),  # left pupil area
            area(uniform(2, 8)),  # right pupil area
            uniform(1, 10),  # pixelsPerDegreeX
            uniform(1, 10)  # pixelsPerDegreeY
        ]


def eye_tracker_server() -> LslDataServer:
    """Create a demo lsl_server that serves eye tracking data.

    Returns:
        LslDataServer: An LslDataServer instance configured for eye tracking data.
    """

    return LslDataServer(device_spec=eye_tracker_device(),
                         generator=eye_tracker_data_generator())


def main():
    """Create and run an lsl_server.

    This function initializes and starts an LSL data server for eye tracking.
    It runs indefinitely until a KeyboardInterrupt is received, at which point
    the server is stopped.
    """
    try:
        server = eye_tracker_server()
        log.info("New server created")
        server.start()
        log.info("Server started")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        server.stop()


if __name__ == '__main__':
    main()
