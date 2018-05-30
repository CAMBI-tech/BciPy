import os
import wx
import wx.lib.agw.gradientbutton as GB
import wx.lib.agw.aquabutton as AB
import wx.lib.buttons as buttons


class BCIGui(wx.Frame):
    """BCIGui."""

    def __init__(self, title: str, size: tuple,
                 background_color: str='blue', parent: wx.Frame=None):
        """Init."""
        super(BCIGui, self).__init__(
            parent, title=title, size=size, style=wx.SYSTEM_MENU | wx.CAPTION | wx.CLOSE_BOX)

        self.panel = wx.Panel(self)
        self.SetBackgroundColour(background_color)

        self.buttons = []
        self.input_text = []
        self.static_text = []
        self.images = []

    def show_gui(self):
        """Show GUI."""
        self.Show(True)

    def close_gui(self):
        pass

    def add_button(self, message: str, position: tuple, size: tuple,
                   button_type: str=None, color: str=None,
                   action: str='default') -> None:
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

    def bind_action(self, action: str, btn: wx.Button) -> None:
        if action == 'default':
            self.Bind(wx.EVT_BUTTON, self.on_clicked, btn)

    def add_text_input(self, position: tuple, size: tuple) -> None:
        """Add Text Input."""
        input_text = wx.TextCtrl(self.panel, pos=position, size=size)
        self.input_text.append(input_text)

    def add_static_text(self, text: str, position: str,
                        color: str, size: int,
                        font_family: wx.Font=wx.FONTFAMILY_SWISS) -> None:
        """Add Text."""

        static_text = wx.StaticText(
            self.panel, pos=position,
            label=text)
        static_text.SetForegroundColour(color)
        font = wx.Font(size, font_family, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_LIGHT)
        static_text.SetFont(font)

        self.static_text.append(static_text)

    def add_image(self, path: str, position: tuple, size: int) -> None:
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

    def on_clicked(self, event):
        """on_clicked.

        Default event to bind to buttons
        """
        btn = event.GetEventObject().GetLabel()
        print(f'pressed {btn}')



if __name__ == '__main__':
    app = wx.App(False)
    gui = BCIGui(title="BCIGui", size=(650, 650), background_color='black')
    gui.show_gui()
    app.MainLoop()
