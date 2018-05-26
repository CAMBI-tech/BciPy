import wx
import wx.lib.scrolledpanel as scrolled
import json
from gui.gui_main import BCIGui
import logging


def font(size: int = 14, font_family: wx.Font=wx.FONTFAMILY_SWISS):
    return wx.Font(size, font_family, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_LIGHT)


def static_text_control(parent, label: str,
                        color: str = 'black', size: int = 14,
                        font_family: wx.Font=wx.FONTFAMILY_SWISS) -> None:
    """Add Text."""

    static_text = wx.StaticText(parent, label=label)
    static_text.SetForegroundColour(color)
    static_text.SetFont(font(size, font_family))
    return static_text


class Form(wx.Panel):
    """The Form class is a wx.Panel that creates controls/inputs for each
    parameter in the provided json file."""

    def __init__(self, parent, json_file='parameters/parameters.json',
                 control_width=300, control_height=25, **kwargs):
        super(Form, self).__init__(parent, **kwargs)

        self.json_file = json_file
        self.control_size = (control_width, control_height)

        with open(json_file) as f:
            data = f.read()

        self.params = json.loads(data)

        # TODO: group inputs by section

        self.createControls()
        self.bindEvents()
        self.doLayout()

    def createControls(self):
        """Create controls (inputs, labels, etc) for each item in the
        parameters file."""

        self.controls = {}
        for key, param in self.params.items():
            if param['type'] == "bool":
                ctl = wx.CheckBox(self, label=param['readableName'])
                ctl.SetValue(param['value'] == 'true')
                self.controls[key] = ctl
            elif type(param['recommended_values']) == list:
                self.controls[f"{key}_label"] = static_text_control(
                    self,
                    label=param['readableName'])
                self.controls[key] = wx.ComboBox(self, size=self.control_size,
                                                 choices=param['recommended_values'],
                                                 style=wx.CB_DROPDOWN)
            # TODO: from wx.lib.masked import NumCtrl
            # elif param['type'] in ['float', 'int']:
            # return numeric_input(key, param)
            else:
                self.controls[f"{key}_label"] = static_text_control(self, label=param['readableName'])
                self.controls[key] = wx.TextCtrl(
                    self, size=self.control_size, value=param['value'])

            self.controls[key].SetFont(font())
        self.saveButton = wx.Button(self, label="Save")

    def bindEvents(self):
        control_events = []
        for k, control in self.controls.items():
            if k in self.params:
                param = self.params[k]
                if param['type'] == "bool":
                    control.Bind(wx.EVT_CHECKBOX, self.checkboxEventHandler(k))
                elif type(param['recommended_values']) == list:
                    control.Bind(wx.EVT_COMBOBOX, self.selectEventHandler(k))
                else:
                    control.Bind(wx.EVT_TEXT, self.textEventHandler(k))

        self.saveButton.Bind(wx.EVT_BUTTON, self.onSave)

    def doLayout(self):
        ''' Layout the controls by means of sizers. '''

        sizer = wx.BoxSizer(wx.VERTICAL)

        # A GridSizer will contain the other controls:
        gridSizer = wx.FlexGridSizer(rows=len(self.controls.keys()) + 1, cols=1,
                                     vgap=10, hgap=10)

        # Prepare some reusable arguments for calling sizer.Add():
        expandOption = dict(flag=wx.EXPAND)
        noOptions = dict()
        emptySpace = ((0, 0), noOptions)

        controls = [(v, noOptions)
                    for k, v in self.controls.items()]

        # Add the controls to the grid:
        for control, options in controls:
            # TODO: add emptySpace after label, control pairs
            gridSizer.Add(control, **options)

        # Add the save button
        gridSizer.Add(self.saveButton, flag=wx.ALIGN_CENTER)

        sizer.Add(gridSizer, border=10, flag=wx.ALL | wx.ALIGN_CENTER)
        self.SetSizerAndFit(sizer)

    # Callback methods:
    def onSave(self, event):
        logging.debug("Saving data")
        with open(self.json_file, 'w') as outfile:
            json.dump(self.params, outfile, indent=4)

        dialog = wx.MessageDialog(
            self, "Parameters Successfully Updated!", 'Info',
            wx.OK | wx.ICON_INFORMATION)
        dialog.ShowModal()
        dialog.Destroy()

    def textEventHandler(self, key):
        def handler(event):
            self.params[key]["value"] = event.GetString()
            logging.debug(f"{key}: {event.GetString()}")
        return handler

    def selectEventHandler(self, key):
        def handler(event):
            self.params[key]["value"] = event.GetString()
            logging.debug(f"{key}: {event.GetString()}")
        return handler

    def checkboxEventHandler(self, key):
        def handler(event):
            self.params[key]["value"] = "true" if event.IsChecked() else "false"
            logging.debug(f"{key}: {bool(event.IsChecked())}")
        return handler


class ScrollPanel(scrolled.ScrolledPanel):
    """Panel which contains the Form. Responsible for handling scrolling."""

    def __init__(self, parent, title="BCI Parameters",
                 json_file="parameters/parameters.json"):
        super(ScrollPanel, self).__init__(parent, -1)

        vbox = wx.BoxSizer(wx.VERTICAL)

        form = Form(self, json_file)
        vbox.Add(static_text_control(self, label=title, size=20))
        vbox.Add(form)
        self.SetSizer(vbox)
        self.SetupScrolling()

    def OnChildFocus(self, event):
        event.Skip()

def main():
    app = wx.App(0)
    frame = wx.Frame(None, wx.ID_ANY,  size=(650, 550), title='BCI Parameters')
    fa = ScrollPanel(frame, title='BCI Parameters',
                     json_file="parameters/parameters.json")
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
