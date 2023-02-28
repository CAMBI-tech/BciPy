"""Demo for the task bar component"""
import time

from psychopy import visual

from bcipy.display.components.task_bar import CalibrationTaskBar, CopyPhraseTaskBar


def make_window():
    """Make a sample window on which to draw."""
    return visual.Window(size=[500, 500],
                         fullscr=False,
                         winType='pyglet',
                         units='norm',
                         waitBlanking=False,
                         color='black')


def demo_calibration(seconds: int = 30):
    """Demo calibration task bar."""
    try:
        win = make_window()
        task_bar = CalibrationTaskBar(win, inquiry_count=100, current_index=1)
        task_bar.draw()
        win.flip()
        print(
            f'Displaying window for {seconds}s... (Interrupt [Ctl-C] to stop)\n'
        )

        time.sleep(2)
        task_bar.update()
        task_bar.draw()
        win.flip()

        while True:
            time.sleep(seconds)
            break
    except KeyboardInterrupt:
        print('Keyboard Interrupt: Demo stopped')
    except Exception as e:
        print(f'{e}')
        raise e
    finally:
        win.close()
        print('Demo complete.')


def demo_copy_phrase(seconds: int = 30):
    """Demo copy phrase task bar."""
    try:
        win = make_window()
        task_bar = CopyPhraseTaskBar(win, spelled_text='HELLO')

        task_bar.draw()
        win.flip()
        print(
            f'Displaying window for {seconds}s... (Interrupt [Ctl-C] to stop)\n'
        )

        time.sleep(1)
        task_bar.update("HELLO_")
        task_bar.draw()
        win.flip()

        while True:
            time.sleep(seconds)
            break
    except KeyboardInterrupt:
        print('Keyboard Interrupt: Demo stopped')
    except Exception as e:
        print(f'{e}')
        raise e
    finally:
        win.close()
        print('Demo complete.')


if __name__ == '__main__':
    # demo_copy_phrase()
    demo_calibration()
