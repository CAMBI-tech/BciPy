"""Sample script to demonstrate usage of the LSL Recorder for writing data from an LSL stream."""

from bcipy.acquisition.protocols.lsl.lsl_recorder import LslRecorder
import time

SLEEP = 10
PATH = '.'

try:
    recorder = LslRecorder(path=PATH)
    recorder.start()
    print(f'\nCollecting data for {SLEEP}s to path=[{PATH}]... (Interrupt [Ctl-C] to stop)\n')

    while True:
        time.sleep(SLEEP)
        recorder.stop()
        break
except IOError as e:
    print(f'{e.strerror}; make sure you started the LSL app or server.')
except KeyboardInterrupt:
    print('Keyboard Interrupt\n')
    recorder.stop()
    print('Stopped')
except Exception as e:
    print(f'{e}')
    raise e
finally:
    print('Stopped')
