import soundfile as sf
from psychopy import core

from bcipy.config import DEFAULT_PARAMETERS_PATH
from bcipy.feedback.sound.auditory_feedback import AuditoryFeedback
from bcipy.helpers.clock import Clock
from bcipy.io.load import load_json_parameters

# Load a parameters file
parameters = load_json_parameters(DEFAULT_PARAMETERS_PATH, value_cast=True)
clock = Clock()

# Init the sound object and give it some time to buffer
try:
    data, fs = sf.read('bcipy/static/sounds/1k_800mV_20ms_stereo.wav',
                       dtype='float32')
    core.wait(1)

except Exception as e:
    print(e)
# Start Auditory Feedback
auditory_feedback = AuditoryFeedback(parameters=parameters, clock=clock)
timing = auditory_feedback.administer(data, fs)

print(timing)
print(auditory_feedback._type())
