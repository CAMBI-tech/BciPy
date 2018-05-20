import subprocess
import os
import wx
import wx.lib.agw.gradientbutton as GB
import wx.lib.agw.aquabutton as AB
import wx.lib.buttons as buttons


class BCIGui(wx.Frame):
    """BCIGui."""

    def __init__(self, title, size,
                 background_color='blue', parameters=None, parent=None):
        """Init."""
        super(BCIGui, self).__init__(parent, title=title, size=size, style=wx.SYSTEM_MENU | wx.CAPTION | wx.CLOSE_BOX)
        self.size = size
        self.panel = wx.Panel(self)
        self.SetBackgroundColour(background_color)

        self.parameters = parameters

        self.buttons = []
        self.input_text = []
        self.static_text = []

    def show_gui(self):
        """Show GUI."""
        self.Show(True)

    def add_button(self, message, position, size,
                   button_type=None, color=None, action='default'):
        """Add Button."""
        # Button Type
        if button_type == 'gradient_button':
            btn = GB.GradientButton(
                self.panel, label=message, pos=position, size=size)
        elif button_type == 'aqua_button':
            btn = AB.AquaButton(
                self.panel, label=message, pos=position, size=size)
        else:
            btn = buttons.GenButton(
                self.panel, label=message, pos=position, size=size)

            # You can really only set colors with GenButtons as the others
            #  use native widgets!
            if color:
                btn.SetBackgroundColour(color)

        # Attach Custom Actions
        if action == 'launch_bci':
            self.Bind(wx.EVT_BUTTON, self.OnClicked, btn)
        elif action == 'default':
            self.Bind(wx.EVT_BUTTON, self.launch_bci_main, btn)

        self.buttons.append(btn)

    def add_text_input(self, position, size):
        """Add Text Input."""
        input_text = wx.TextCtrl(self.panel, pos=position, size=size)
        self.input_text.append(input_text)

    def add_static_text(self, text, position, color, size):
        """Add Text."""

        static_text = wx.StaticText(
            self.panel, pos=position,
            label=text)
        static_text.SetForegroundColour(color)
        font = wx.Font(size, wx.FONTFAMILY_MODERN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_LIGHT)
        static_text.SetFont(font)

        self.static_text.append(static_text)

    def add_image(self, path, position, size):
        """Add Image."""
        if os.path.isfile(path):
            img = wx.Image(path, wx.BITMAP_TYPE_ANY)
            # scale the image, preserving the aspect ratio
            W = img.GetWidth()
            H = img.GetHeight()
            if W > H:
                NewW = size
                NewH = size * H / W
            else:
                NewH = size
                NewW = size * W / H

            img = img.Scale(NewW, NewH)
            img = img.ConvertToBitmap()
            bmp = wx.StaticBitmap(self.panel, pos=position, bitmap=img)

        else:
            print('INVALID PATH')

    def add_scroll(self):
        """Add Scroll."""
        pass

    def OnClicked(self, event):
        """OnClicked."""
        event.GetEventObject().GetLabel()

        # print(f'pressed {btn}')

    def launch_bci_main(self, event):
        """Laucnh BCI MAIN"""
        if self.check_input():
            username = self.input_text[0].GetValue().replace(" ", "_")
            experiment_type = _cast_experiment_type(
                event.GetEventObject().GetLabel())
            mode = 'RSVP'
            cmd = 'python bci_main.py -m {} -t {} -u {}'.format(
                mode, experiment_type, username)

            subprocess.call(cmd, shell=True)

    def check_input(self):
        """Check Input."""
        if self.input_text[0].GetValue() == '':
            dialog = wx.MessageDialog(
                self, "Please Input User ID", 'Info', wx.OK | wx.ICON_WARNING)
            dialog.ShowModal()
            dialog.Destroy()
            return False
        return True


def _cast_experiment_type(experiment_type_string):
    if experiment_type_string == 'Calibration':
        experiment_type = 1
    elif experiment_type_string == 'Copy Phrase':
        experiment_type = 2
    elif experiment_type_string == 'Copy Phrase Calibration':
        experiment_type = 3
    else:
        raise ValueError('Not a known experiment_type')

    return experiment_type


if __name__ == '__main__':
    app = wx.App(False)
    gui = BCIGui(title="BCIGui", size=(650, 650), background_color='black')
    gui.show_gui()
    app.MainLoop()
