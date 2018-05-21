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
        super(BCIGui, self).__init__(
            parent, title=title, size=size, style=wx.SYSTEM_MENU | wx.CAPTION | wx.CLOSE_BOX)
        self.size = size
        self.panel = wx.Panel(self)
        self.SetBackgroundColour(background_color)

        self.parameters = parameters

        self.buttons = []
        self.input_text = []
        self.static_text = []
        self.images = []

    def show_gui(self):
        """Show GUI."""
        self.Show(True)

    def close_gui(self):
        pass

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
        self.bind_action(action, btn)

        self.buttons.append(btn)

    def bind_action(self, action, btn):
        if action == 'default':
            self.Bind(wx.EVT_BUTTON, self.on_clicked, btn)

    def add_text_input(self, position, size):
        """Add Text Input."""
        input_text = wx.TextCtrl(self.panel, pos=position, size=size)
        self.input_text.append(input_text)

    def add_static_text(self, text, position, color, size, font_family=wx.FONTFAMILY_SWISS):
        """Add Text."""

        static_text = wx.StaticText(
            self.panel, pos=position,
            label=text)
        static_text.SetForegroundColour(color)
        font = wx.Font(size, font_family, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_LIGHT)
        static_text.SetFont(font)

        self.static_text.append(static_text)

    def add_image(self, path, position, size):
        """Add Image."""
        if os.path.isfile(path):
            img = wx.Image(path, wx.BITMAP_TYPE_ANY)
            # scale the image, preserving the aspect ratio
            width = img.GetWidth()
            height = img.GetHeight()

            if width > height:
                new_width = size
                new_height = size * height / width
            else:
                new_height = size
                new_width = size * width / height

            img = img.Scale(new_width, new_height)
            img = img.ConvertToBitmap()
            bmp = wx.StaticBitmap(self.panel, pos=position, bitmap=img)

            self.images.append(bmp)

        else:
            print('INVALID PATH')

    def add_scroll(self):
        """Add Scroll."""
        pass

    def on_clicked(self, event):
        """on_clicked."""
        btn = event.GetEventObject().GetLabel()
        print(f'pressed {btn}')

    def edit_parameters(self, event):
        """Edit Parameters.

        Function for executing the edit parameter window
        """
        print(f'edit parameters window launched')

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

    def launch_mode(self, event):
        if self.check_input():
            mode = _cast_mode(
                event.GetEventObject().GetLabel())
            cmd = 'python bci_main.py -m {} -t {} -u {}'.format(
                mode, experiment_type, username)

            subprocess.call(cmd, shell=True)

    def check_input(self):
        """Check Input."""
        try:
            if self.input_text[0].GetValue() == '':
                dialog = wx.MessageDialog(
                    self, "Please Input User ID", 'Info', wx.OK | wx.ICON_WARNING)
                dialog.ShowModal()
                dialog.Destroy()
                return False
        except:
            dialog = wx.MessageDialog(
                self, "Error, expected input field for this function", 'Info', wx.OK | wx.ICON_WARNING)
            dialog.ShowModal()
            dialog.Destroy()
            return False
        return True


def _cast_experiment_type(experiment_type_string):
    if experiment_type_string == 'Calibration':
        experiment_type = 1
    elif experiment_type_string == 'Copy Phrase':
        experiment_type = 2
    elif experiment_type_string == 'Copy Phrase C.':
        experiment_type = 3
    else:
        raise ValueError('Not a known experiment_type')

    return experiment_type


if __name__ == '__main__':
    app = wx.App(False)
    gui = BCIGui(title="BCIGui", size=(650, 650), background_color='black')
    gui.show_gui()
    app.MainLoop()
