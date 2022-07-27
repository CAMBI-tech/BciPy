from bcipy.acquisition.protocols.lsl.lsl_recorder import LslRecorder
import time

SLEEP = 10
recorder = LslRecorder(path='.')

try:
    recorder.start()
    print(f'\nCollecting data for {SLEEP}s... (Interrupt [Ctl-C] to stop)\n')

    while True:
        time.sleep(SLEEP)
        recorder.stop()
        break
except IOError as e:
    print(f'{e.strerror}; make sure you started the LSL app or server.')
except KeyboardInterrupt:
    print('Keyboard Interrupt')
    recorder.stop()
    print('Stopped')
except Exception as e:
    print(f'{e}')
    recorder.stop()
    print('Stopped')
    raise e
