import sys, os, psutil, pyglet

from os.path import join, getsize
for root, dirs, files in os.walk('python/Lib/email'):
    print root, "consumes",
    print sum(getsize(join(root, name)) for name in files),
    print "bytes in", len(files), "non-directory files"
    if 'CVS' in dirs:
        dirs.remove('CVS')  # don't visit CVS directories



def get_system_info():


    # Three lines for getting screen resolution
    platform = pyglet.window.get_platform()
    display = platform.get_default_display()
    screen = display.get_default_screen()

    mem = psutil.virtual_memory()

    return {
        'OS': sys.platform,
        'PYTHON': sys.version,
        'RESOLUTION': [screen.width, screen.height],
        'PYTHONPATH': os.environ['PYTHONPATH'].split(os.pathsep),
        'AVAILMEMORYMB': mem.available/1024./1024
    }