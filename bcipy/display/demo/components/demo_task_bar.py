"""Demo for the task bar component"""
import time

from psychopy import visual

from bcipy.display.components.task_bar import TaskBar


def make_window():
    """Make a sample window on which to draw"""
    return visual.Window(size=[500, 500],
                         fullscr=False,
                         winType='pyglet',
                         units='norm',
                         waitBlanking=False,
                         color='black')

def main(seconds: int=100):
    try:
        win = make_window()
        task_bar = TaskBar(win)

        task_bar.draw()
        win.flip()
        print(f'Displaying window for {seconds}s... (Interrupt [Ctl-C] to stop)\n')

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
    main()
