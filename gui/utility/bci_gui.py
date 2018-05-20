import subprocess
import wx
import wx.lib.agw.gradientbutton as GB
import wx.lib.agw.aquabutton as AB


class BCIGui(wx.Frame):
    """docstring for BCIGui"""
    def __init__(self, title, size, parameters=None, parent=None):
        super(BCIGui, self).__init__(parent, title=title, size=size)
        self.parameters = parameters
        self.panel = wx.Panel(self)

        self.buttons = []

    def show_gui(self):
        self.Show(True)

    def add_button(self, message, position, size):
        btn = GB.GradientButton(
            self.panel, label=message, pos=position, size=size)
        btn.Bind(wx.EVT_BUTTON, self.OnClicked)

    def add_text_input(self, message, position, size):
        pass

    def add_window(self):
        pass

    def add_text(self, xpos, ypos, color, size, text):
        pass

    def add_image(self):
        pass

    def add_scroll(self):
        pass

    def OnClicked(self, event):
        btn = event.GetEventObject().GetLabel()
        print(f'pressed {btn}')


if __name__ == '__main__':
    app = wx.App(False) 
    gui = BCIGui("BCIGui", (500,500))
    gui.add_button(message="Test Button", position=(0,0), size=(80, 80))
    gui.add_button(message="Test Button 2", position=(80,0), size=(80, 80))
    gui.show_gui()
    app.MainLoop()
