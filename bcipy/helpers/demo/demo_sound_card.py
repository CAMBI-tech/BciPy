'''demo for using sound device and sound file:
    Taken from: https://python-sounddevice.readthedocs.io/en/0.2.1/examples.html
'''

import argparse
import logging
log = logging.getLogger(__name__)
# To use, cd into helpers directory, run >> python demo/sound_card_demo.py "filename"
# Example: python demo/sound_card_demo.py "../static/sounds/chime.wav"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("filename", help="audio file to be played back")
parser.add_argument("-d", "--device", type=int, help="device ID")
args = parser.parse_args()

try:
    import sounddevice as sd
    import soundfile as sf
    devices = sd.query_devices()
    print(devices)
    data, fs = sf.read(args.filename, dtype='float32')
    sd.play(data, fs, device=args.device, blocking=True)
    status = sd.get_status()
    if status:
        log.warning(str(status))
except BaseException as e:
    # This avoids printing the traceback, especially if Ctrl-C is used.
    raise SystemExit(str(e))
