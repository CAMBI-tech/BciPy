import sys, os, psutil, pyglet


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
        'AVAILMEMORYMB': mem.available/1024./1024
    }