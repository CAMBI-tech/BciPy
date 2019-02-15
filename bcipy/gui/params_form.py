import wx
import wx.lib.scrolledpanel as scrolled
import json
import logging
from typing import Callable, Dict, Tuple
from collections import namedtuple
from os import sep

log = logging.getLogger(__name__)
JSON_INDENT = 2


# Utility functions
def font(size: int = 14, font_family: wx.Font=wx.FONTFAMILY_SWISS) -> wx.Font:
    """Create a Font object with the given parameters."""
    return wx.Font(size, font_family, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_LIGHT)


def static_text_control(parent, label: str,
                        color: str = 'black', size: int = 14,
                        font_family: wx.Font=wx.FONTFAMILY_SWISS) -> wx.StaticText:
    """Creates a static text control with the given font parameters. Useful for
    creating labels and help components."""

    static_text = wx.StaticText(parent, label=label)
    static_text.SetForegroundColour(color)
    static_text.SetFont(font(size, font_family))
    return static_text


# Holds the label, help, and input controls for a given parameter.
FormInput = namedtuple('FormInput', ['control', 'label', 'help'])
Parameter = namedtuple('Parameter', ['value', 'section', 'readableName',
                                     'helpTip', 'recommended_values', 'type'])


class Form(wx.Panel):
    """The Form class is a wx.Panel that creates controls/inputs for each
    parameter in the provided json file."""

    def __init__(self, parent,
                 json_file: str='bcipy/parameters/parameters.json',
                 load_file: str=None,
                 control_width: int=300, control_height: int=25, **kwargs):
        super(Form, self).__init__(parent, **kwargs)

        self.json_file = json_file
        self.load_file = json_file if not load_file else load_file
        self.control_size = (control_width, control_height)
        self.help_font_size = 12
        self.help_color = 'DARK SLATE GREY'
        with open(self.load_file) as f:
            data = f.read()

        config = json.loads(data)
        self.params = {k: Parameter(*v.values()) for k, v in config.items()}

        # TODO: group inputs by section

        self.createControls()
        self.bindEvents()
        self.doLayout()

    def createControls(self):
        """Create controls (inputs, labels, etc) for each item in the
        parameters file."""

        self.controls = {}
        for key, param in self.params.items():
            if param.type == "bool":
                form_input = self.bool_input(param)
            elif "path" in param.type:
                form_input = self.file_input(param)
            elif type(param.recommended_values) == list:
                form_input = self.selection_input(param)
            # TODO: NumCtrl for numeric input types.
            # from wx.lib.masked import NumCtrl
            # elif param['type'] in ['float', 'int']:
            else:
                form_input = self.text_input(param)

            self.add_input(self.controls, key, form_input)

        self.saveButton = wx.Button(self, label="Save")

    def add_input(self, controls: Dict[str, wx.Control], key: str,
                  form_input: FormInput) -> None:
        """Adds the controls for the given input to the controls structure."""
        if form_input.label:
            controls[f"{key}_label"] = form_input.label
        if form_input.help:
            controls[f"{key}_help"] = form_input.help
        controls[key] = form_input.control

        # TODO: consider adding an empty space after each input:
        # controls[f"{key}_empty"] = (0,0)

    def bool_input(self, param: Parameter) -> FormInput:
        """Creates a checkbox FormInput"""
        ctl = wx.CheckBox(self, label=param.readableName)
        ctl.SetValue(param.value == 'true')
        ctl.SetFont(font())
        return FormInput(ctl, label=None, help=None)

    def selection_input(self, param: Parameter) -> FormInput:
        """Creates a selection pulldown FormInput."""
        ctl = wx.ComboBox(self, -1, param.value,
                          size=self.control_size,
                          choices=param.recommended_values,
                          style=wx.CB_DROPDOWN)
        ctl.SetFont(font())
        label, help_tip = self.input_label(param)
        return FormInput(ctl, label, help_tip)

    def text_input(self, param: Parameter) -> FormInput:
        """Creates a text field FormInput."""
        ctl = wx.TextCtrl(self, size=self.control_size, value=param.value)
        ctl.SetFont(font())
        label, help_tip = self.input_label(param)
        return FormInput(ctl, label, help_tip)
        
    def file_input(self, param: Parameter) -> FormInput:
        """Creates a text field or selection pulldown FormInput with a button 
        to browse for a file."""
        #Creates a combobox instead of text field if the parameter has recommended values
        if type(param.recommended_values) == list:
            ctl = wx.ComboBox(self, -1, param.value,
                size=self.control_size,
                choices=param.recommended_values,
                style=wx.CB_DROPDOWN)   
        else:
            ctl = wx.TextCtrl(self, size=self.control_size, value=param.value)
        ctl.SetFont(font())
        btn = wx.Button(self, label="...", size=(self.control_size[1], self.control_size[1]))
        ctl_array = [ctl, btn]
        label, help_tip = self.input_label(param)
        return FormInput(ctl_array, label, help_tip)

    def input_label(self, param: Parameter) -> Tuple[wx.Control, wx.Control]:
        """Returns a label control and maybe a help Control if the help
        text is different than the label text."""
        label = static_text_control(self, label=param.readableName)
        help_tip = None
        if param.readableName != param.helpTip:
            help_tip = static_text_control(self, label=param.helpTip,
                                           size=self.help_font_size,
                                           color=self.help_color)
        return (label, help_tip)

    def bindEvents(self):
        """Bind event handlers to the controls to update the parameter values
        when the inputs change."""

        control_events = []
        for k, control in self.controls.items():
            # Only bind events for control inputs, not for label and help items
            if k in self.params:
                param = self.params[k]
                if param.type == "bool":
                    control.Bind(wx.EVT_CHECKBOX, self.checkboxEventHandler(k))
                elif "path" in param.type:
                    is_directory = False
                    if param.type == "directorypath":
                        is_directory = True
                    control[0].Bind(wx.EVT_TEXT, self.textEventHandler(k))
                    control[1].Bind(wx.EVT_BUTTON, self.buttonEventHandler(k, is_directory))
                elif type(param.recommended_values) == list:
                    control.Bind(wx.EVT_COMBOBOX, self.selectEventHandler(k))
                    # allow user to input a value not in the select list.
                    control.Bind(wx.EVT_TEXT, self.textEventHandler(k))
                else:
                    control.Bind(wx.EVT_TEXT, self.textEventHandler(k))

        self.saveButton.Bind(wx.EVT_BUTTON, self.onSave)

    def doLayout(self):
        """Layout the controls using Sizers."""

        sizer = wx.BoxSizer(wx.VERTICAL)

        button_control_len = 1  # save button
        rowlen = len(self.controls.keys()) + button_control_len
        gridSizer = wx.FlexGridSizer(rows=rowlen, cols=1, vgap=10, hgap=10)

        noOptions = dict()

        # Add the controls to the grid:
        for control in self.controls.values():
            if isinstance(control, list):
                hbox = wx.BoxSizer(wx.HORIZONTAL)
                hbox.Add(control[0], flag=wx.RIGHT, border=5)
                hbox.Add(control[1], **noOptions)
                gridSizer.Add(hbox, **noOptions)
            else:
                gridSizer.Add(control, **noOptions)

        # Add the save button
        gridSizer.Add(self.saveButton, flag=wx.ALIGN_CENTER)

        sizer.Add(gridSizer, border=10, flag=wx.ALL | wx.ALIGN_CENTER)
        self.SetSizerAndFit(sizer)

    # Callback methods:
    def onSave(self, event: wx.EVT_BUTTON) -> None:
        log.debug("Saving parameter data")

        with open(self.json_file, 'w') as outfile:
            json.dump({k: v._asdict() for k, v in self.params.items()},
                      outfile, indent=JSON_INDENT)

        dialog = wx.MessageDialog(
            self, "Parameters Successfully Updated!", 'Info',
            wx.OK | wx.ICON_INFORMATION)
        dialog.ShowModal()
        dialog.Destroy()

    def textEventHandler(self, key: str) -> Callable[[wx.EVT_TEXT], None]:
        """Returns a handler function that updates the parameter for the
        provided key.
        """
        def handler(event: wx.EVT_TEXT):
            p = self.params[key]
            self.params[key] = p._replace(value=event.GetString())
            log.debug(f"{key}: {event.GetString()}")
        return handler

    def selectEventHandler(self, key: str) -> Callable[[wx.EVT_COMBOBOX], None]:
        """Returns a handler function that updates the parameter for the
        provided key.
        """
        def handler(event: wx.EVT_COMBOBOX):
            p = self.params[key]
            self.params[key] = p._replace(value=event.GetString())
            log.debug(f"{key}: {event.GetString()}")
        return handler

    def checkboxEventHandler(self, key: str) -> Callable[[wx.EVT_CHECKBOX], None]:
        """Returns a handler function that updates the parameter for the
        provided key.
        """
        def handler(event: wx.EVT_CHECKBOX):
            p = self.params[key]
            value = "true" if event.IsChecked() else "false"
            self.params[key] = p._replace(value=value)
            log.debug(f"{key}: {bool(event.IsChecked())}")
        return handler
        
    def buttonEventHandler(self, key: str, directory: bool) -> Callable[[wx.EVT_BUTTON], None]:
        """Returns a handler function that updates the parameter for the
        provided key.
        """
        def handler(event: wx.EVT_BUTTON):
            #Change dialog type depending on whether parameter requires a directory or file
            if directory:
                file_dialog = wx.DirDialog(self, "Select a path", style=wx.FD_OPEN)
            else: 
                file_dialog = wx.FileDialog(self, "Select a file", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
                
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                new_path = None
            else:
                new_path = str(file_dialog.GetPath())
                if directory:
                    #Add operating system separator character to end of directory path
                    new_path = new_path + sep
                log.debug(new_path)
                    
            if new_path:
                for item_key, field in self.controls.items():
                    if item_key == key:
                        field[0].SetValue(new_path)                            
                parameter_key = self.params[key]
                self.params[key] = parameter_key._replace(value=new_path)
                log.debug(f"Selected path: {new_path}")
                
        return handler


class MainPanel(scrolled.ScrolledPanel):
    """Panel which contains the Form. Responsible for selecting the json data
     to load and handling scrolling."""

    def __init__(self, parent, title="BCI Parameters",
                 json_file="bcipy/parameters/parameters.json"):
        super(MainPanel, self).__init__(parent, -1)
        self.json_file = json_file
        vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox = vbox

        self.form = Form(self, json_file)
        vbox.Add(static_text_control(self, label=title, size=20),
                 0, wx.TOP | wx.ALIGN_CENTER_HORIZONTAL, border=5)
        vbox.AddSpacer(10)

        loading_box = wx.BoxSizer(wx.VERTICAL)
        loading_box.Add(static_text_control(self, label=f"Editing: {json_file}",
                                            size=14))
        self.loaded_from = static_text_control(
            self,
            label=f"Loaded from: {json_file}",
            size=14)
        loading_box.Add(self.loaded_from)
        loading_box.AddSpacer(10)

        self.loadButton = wx.Button(self, label="Load")
        self.loadButton.Bind(wx.EVT_BUTTON, self.onLoad)
        loading_box.Add(self.loadButton)

        # Used for displaying help messages to the user.
        self.flash_msg = static_text_control(self, label='', size=14,
                                             color='FOREST GREEN')
        loading_box.Add(self.flash_msg)

        vbox.Add(loading_box, 0, wx.ALL, border=10)
        vbox.AddSpacer(10)
        vbox.Add(self.form)
        self.SetSizer(vbox)
        self.SetupScrolling()

    def onLoad(self, event: wx.EVT_BUTTON) -> None:
        """Event handler to load the form data from a different json file."""
        log.debug("Loading parameters file")

        with wx.FileDialog(self, "Open parameters file",
                           wildcard="JSON files (*.json)|*.json",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fd:
            if fd.ShowModal() == wx.ID_CANCEL:
                return     # the user changed their mind
            load_file = fd.GetPath()
            self.loaded_from.SetLabel(f"Loaded from: {load_file}")
            self.flash_msg.SetLabel("Click the Save button to persist these "
                                    "changes.")
            self.vbox.Hide(self.form)
        self.form = Form(self, json_file=self.json_file, load_file=load_file)
        self.vbox.Add(self.form)
        self.SetupScrolling()

    def OnChildFocus(self, event):
        event.Skip()


def main(title='BCI Parameters', size=(650, 550),
         json_file="bcipy/parameters/parameters.json"):
    """Set up the GUI components and start the main loop."""

    app = wx.App(0)
    frame = wx.Frame(None, wx.ID_ANY,  size=size, title=title)
    fa = MainPanel(frame, title=title, json_file=json_file)
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()
