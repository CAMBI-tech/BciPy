"""Mock eye tracker that streams data using LabStreamingLayer"""

import logging
import math
import time

from numpy.random import uniform
from pylsl import StreamInfo, StreamOutlet

from bcipy.acquisition.datastream.lsl_server import LslDataServer
from bcipy.acquisition.devices import DeviceSpec
from bcipy.signal.generator.generator import gen_random_data

log = logging.getLogger(__name__)

def eye_tracker_device() -> DeviceSpec:
    return DeviceSpec(name='EyeTracker',
                      channels=[
                          'leftEyeX', 'leftEyeY', 'rightEyeX', 'rightEyeY',
                          'leftPupilArea', 'rightPupilArea',
                          'pixelsPerDegreeX', 'pixelsPerDegreeY'
                      ],
                      sample_rate=500.0,
                      content_type='Gaze')


def eye_tracker_data_generator(display_x=1920, display_y=1080):
    """Generates sample eye tracker data.

    TODO: determine appropriate values for pixelsPerDegree fields.
    TODO: look info alternatives; maybe PyGaze.
    http://www.pygaze.org/about/
    """

    def area(diameter):
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
    """Create a demo lsl_server that serves eye tracking data."""

    return LslDataServer(device_spec=eye_tracker_device(),
                         generator=eye_tracker_data_generator(),
                         include_meta=False)


def main():
    """Create an run an lsl_server"""
    try:
        server = eye_tracker_server()
        log.debug("New server created")
        server.start()
        log.debug("Server started")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        server.stop()


if __name__ == '__main__':
    main()
