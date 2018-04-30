from helpers.load import load_json_parameters
from psychopy import core
from feedback.sound.auditory_feedback import AuditoryFeedback
import soundfile as sf


# Load a parameters file
parameters = load_json_parameters('parameters/parameters.json')
clock = core.Clock()

# Init the sound object and give it some time to buffer
try:
    data, fs = sf.read('./static/sounds/1k_800mV_20ms_stereo.wav',
                       dtype='float32')
    core.wait(1)

except Exception as e:
    print(e)
# Start Auditory Feedback
auditory_feedback = AuditoryFeedback(parameters=parameters, clock=clock)
timing = auditory_feedback.administer(data, fs)

print(timing)
print(auditory_feedback._type())
