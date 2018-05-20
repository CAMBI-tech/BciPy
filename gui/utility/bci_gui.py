import subprocess
import wx
import wx.lib.agw.gradientbutton as GB
import wx.lib.agw.aquabutton as AB
import wx.lib.buttons as buttons


class BCIGui(wx.Frame):
    """BCIGui."""

    def __init__(self, title, size,
                 background_color='blue', parameters=None, parent=None):
        """Init."""
        super(BCIGui, self).__init__(parent, title=title, size=size)
        self.panel = wx.Panel(self)
        self.SetBackgroundColour(background_color)

        self.parameters = parameters

        self.buttons = []
        self.input_text = []

    def show_gui(self):
        """Show GUI."""
        self.Show(True)

    def add_button(self, message, position, size,
                   button_type=None, color=None, action='default'):
        """Add Button."""
        # Button Type
        if button_type == 'gradient_button':
            btn = GB.GradientButton(
                self.panel, label=message.center(5), pos=position, size=size)
        elif button_type == 'aqua_button':
            btn = AB.AquaButton(
                self.panel, label=message.center(5), pos=position, size=size)
        else:
            btn = buttons.GenButton(
                self.panel, label=message.center(5), pos=position, size=size)

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

    def add_window(self):
        """Add Window."""
        pass

    def add_text(self, xpos, ypos, color, size, text):
        """Add Text."""
        pass

    def add_image(self):
        """Add Image."""
        pass

    def add_scroll(self):
        """Add Scroll."""
        pass

    def OnClicked(self, event):
        """OnClicked."""
        btn = event.GetEventObject().GetLabel()
        print(f'pressed {btn}')

    def launch_bci_main(self, event):
        if self.check_input():
            username = self.input_text[0].GetValue()
            experiment_type = _cast_experiment_type(
                event.GetEventObject().GetLabel())
            print(username, experiment_type)

    def check_input(self):
        if self.input_text[0].GetValue() == '':
            print('error!')
            return False
        return True


def _cast_experiment_type(experiment_type_string):
    if experiment_type_string == 'Calibration':
        experiment_type = 1
    elif experiment_type == 'Copy Phrase':
        experiment_type = 2
    elif experiment_type == 'Copy Phrase Calibration':
        experiment_type = 3
    else:
        raise ValueError('Not a known experiment_type')

    return experiment_type


if __name__ == '__main__':
    app = wx.App(False)
    gui = BCIGui(title="BCIGui", size=(650, 650), background_color='black')
    gui.add_button(
        message="Calibration",
        position=(50, 400), size=(100, 100),
        color='red')
    gui.add_button(
        message="Copy Phrase", position=(200, 400),
        size=(100, 100),
        color='blue')
    gui.add_button(
        message="Copy Phrase Calibratui", position=(400, 400),
        size=(100, 100),
        color='blue')
    gui.add_text_input(position=(175, 100), size=(300, 50))
    gui.show_gui()
    app.MainLoop()
