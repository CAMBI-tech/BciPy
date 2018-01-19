from psychopy import visual, core, event
from utils.get_system_info import get_system_info


system_info = get_system_info()
'''
system_info is a dictionary with labels:
        'OS': sys.platform,
        'PYTHON': sys.version,
        'RESOLUTION': [screen.width, screen.height],
        'PYTHONPATH': os.environ['PYTHONPATH'].split(os.pathsep),
        'AVAILMEMORYMB': mem.available/1024./1024
'''


shuffle_window = visual.Window(system_info['RESOLUTION'],
                               monitor='testMonitor', units='deg')

# create some stimuli

grating = visual.GratingStim(win=shuffle_window, mask="circle", size=3, pos=[-4,0], sf=3)
fixation = visual.GratingStim(win=shuffle_window, size=0.5, pos=[0,0], sf=0, color='red')


while True:
    event.clearEvents()
    grating.setPhase(0.05, '+')  # advance phase by 0.05 of a cycle
    grating.draw()
    fixation.draw()
    shuffle_window.update()
    num_keys = len(event.getKeys())
    if num_keys > 0:
        print num_keys
        break
    event.clearEvents()

core.wait(2.0)

shuffle_window.close()
core.quit()