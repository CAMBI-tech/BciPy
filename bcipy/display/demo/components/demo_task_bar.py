"""Demo for the task bar component"""
import time
from typing import Callable
from psychopy import visual

from bcipy.display.components.task_bar import (DEFAULT_TASK_PROPERTIES,
                                               CalibrationTaskBar,
                                               CopyPhraseTaskBar, TaskBar)


def make_window():
    """Make a sample window on which to draw."""
    return visual.Window(size=[500, 500],
                         fullscr=False,
                         winType='pyglet',
                         units='norm',
                         waitBlanking=False,
                         color='black')


def demo_task_bar(win: visual.Window):
    """Demo generic task bar."""
    config = DEFAULT_TASK_PROPERTIES
    config.task_text = 'Task Bar'
    config.task_color[0] = 'magenta'
    task_bar = TaskBar(win, config=config)
    task_bar.draw()
    win.flip()


def demo_calibration(win: visual.Window):
    """Demo calibration task bar."""

    task_bar = CalibrationTaskBar(win, inquiry_count=100, current_index=1)
    task_bar.draw()
    win.flip()
    time.sleep(2)
    task_bar.update()
    task_bar.draw()
    win.flip()


def demo_copy_phrase(win: visual.Window):
    """Demo copy phrase task bar."""
    config = DEFAULT_TASK_PROPERTIES
    config.task_text = 'HELLO_WORLD'
    config.task_color.append('green')

    task_bar = CopyPhraseTaskBar(win, config=config, spelled_text='HELLO')

    task_bar.draw()
    win.flip()

    time.sleep(1)
    task_bar.update("HELLO_")
    task_bar.draw()
    win.flip()


def run(demo: Callable[[visual.Window], None], seconds=30):
    """Run the given function for the provided duration.

    Parameters
    ----------
        demo - function to run; a Window object is passed to this function.
        seconds - Window is closed after this duration.
    """
    try:
        win = make_window()
        print(
            f'Displaying window for {seconds}s... (Interrupt [Ctl-C] to stop)\n'
        )
        demo(win)
        while True:
            time.sleep(seconds)
            break
    except KeyboardInterrupt:
        print('Keyboard Interrupt: Demo stopped')
    except Exception as other_exception:
        print(f'{other_exception}')
        raise other_exception
    finally:
        win.close()
        print('Demo complete.')


if __name__ == '__main__':
    run(demo_calibration)
    # run(demo_copy_phrase)
    # run(demo_task_bar)
