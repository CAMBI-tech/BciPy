import sys, Tkinter, os, psutil


def get_system_info():

    tk = Tkinter.Tk()
    mem = psutil.virtual_memory()

    dict_of_attributes = {
        'OS': sys.platform,
        'PYTHON': sys.version,
        'RESOLUTION': str(tk.winfo_screenwidth()) + 'x' + str(tk.winfo_screenheight()),
        'PYTHONPATH': os.path,
        'AVAILMEMORYMB': mem.available/1024./1024
    }

    return dict_of_attributes