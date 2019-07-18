import wx
import wx.lib.scrolledpanel as scrolled
import json
import logging
from typing import Callable, Dict, Tuple
from collections import namedtuple
from os import sep, path
from bcipy.helpers.load import check_if_parameters_contains_all_keys
from bcipy.helpers.system_utils import bcipy_version

log = logging.getLogger(__name__)
JSON_INDENT = 2
DEFAULT_PARAMETERS_LOCATION = 'bcipy/parameters/parameters.json'


def font(size: int = 14, font_family: wx.Font = wx.FONTFAMILY_SWISS) -> wx.Font:
    """Create a Font object with the given parameters."""
    return wx.Font(size, font_family, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_LIGHT)


def static_text_control(parent,
                        label: str,
                        color: str = 'black',
                        size: int = 14,
                        font_family: wx.Font = wx.FONTFAMILY_SWISS) -> wx.StaticText:
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
                 json_file: str = 'bcipy/parameters/parameters.json',
                 load_file: str = None,
                 control_width: int = 300, control_height: int = 25, **kwargs):
        super(Form, self).__init__(parent, **kwargs)

        self.json_file = json_file
        self.load_file = json_file if not load_file else load_file
        self.control_size = (control_width, control_height)
        self.help_font_size = 12
        self.help_color = 'DARK SLATE GREY'
        self.params = parent.params

        # TODO: group inputs by section

        self.createControls()
        self.bindEvents()
        self.doLayout()

    def createControls(self):
        """Create controls (inputs, labels, etc) for each item in the
        parameters file."""

        self.controls = {}
        for key, param in self.params.items():
            if param.type == 'bool':
                form_input = self.bool_input(param)
            elif 'path' in param.type:
                form_input = self.file_input(param)
            elif isinstance(param.recommended_values, list):
                form_input = self.selection_input(param)
            # TODO: NumCtrl for numeric input types.
            # from wx.lib.masked import NumCtrl
            # elif param['type'] in ['float', 'int']:
            else:
                form_input = self.text_input(param)

            self.add_input(self.controls, key, form_input)

    def add_input(self, controls: Dict[str, wx.Control], key: str,
                  form_input: FormInput) -> None:
        """Adds the controls for the given input to the controls structure."""
        if form_input.label:
            controls[f'{key}_label'] = form_input.label
        if form_input.help:
            controls[f'{key}_help'] = form_input.help
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
        """File Input.

        Creates a text field or selection pulldown FormInput with a button
        to browse for a file.
        """
        # Creates a combobox instead of text field if the parameter has
        # recommended values
        if isinstance(param.recommended_values, list):
            ctl = wx.ComboBox(
                self,
                -1,
                param.value,
                size=self.control_size,
                choices=param.recommended_values,
                style=wx.CB_DROPDOWN)
        else:
            ctl = wx.TextCtrl(self, size=self.control_size, value=param.value)
        ctl.SetFont(font())
        btn = wx.Button(
            self,
            label='...',
            size=(
                self.control_size[1],
                self.control_size[1]))
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
                if param.type == 'bool':
                    control.Bind(wx.EVT_CHECKBOX, self.checkboxEventHandler(k))
                elif 'path' in param.type:
                    is_directory = False
                    if param.type == 'directorypath':
                        is_directory = True
                    control[0].Bind(wx.EVT_TEXT, self.textEventHandler(k))
                    control[1].Bind(
                        wx.EVT_BUTTON, self.buttonEventHandler(
                            k, is_directory))
                elif isinstance(param.recommended_values, list):
                    control.Bind(wx.EVT_COMBOBOX, self.selectEventHandler(k))
                    # allow user to input a value not in the select list.
                    control.Bind(wx.EVT_TEXT, self.textEventHandler(k))
                else:
                    control.Bind(wx.EVT_TEXT, self.textEventHandler(k))

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

        sizer.Add(gridSizer, border=10, flag=wx.ALL | wx.ALIGN_CENTER)
        self.SetSizerAndFit(sizer)

    # Callback methods:
    def textEventHandler(self, key: str) -> Callable[[wx.EVT_TEXT], None]:
        """Returns a handler function that updates the parameter for the
        provided key.
        """
        def handler(event: wx.EVT_TEXT):
            p = self.params[key]
            self.params[key] = p._replace(value=event.GetString())
            log.debug(f"{key}: {event.GetString()}")
        return handler

    def selectEventHandler(
            self, key: str) -> Callable[[wx.EVT_COMBOBOX], None]:
        """Returns a handler function that updates the parameter for the
        provided key.
        """
        def handler(event: wx.EVT_COMBOBOX):
            p = self.params[key]
            self.params[key] = p._replace(value=event.GetString())
            log.debug(f"{key}: {event.GetString()}")
        return handler

    def checkboxEventHandler(
            self, key: str) -> Callable[[wx.EVT_CHECKBOX], None]:
        """Returns a handler function that updates the parameter for the
        provided key.
        """
        def handler(event: wx.EVT_CHECKBOX):
            p = self.params[key]
            value = 'true' if event.IsChecked() else 'false'
            self.params[key] = p._replace(value=value)
            log.debug(f'{key}: {bool(event.IsChecked())}')
        return handler

    def buttonEventHandler(
            self, key: str, directory: bool) -> Callable[[wx.EVT_BUTTON], None]:
        """Returns a handler function that updates the parameter for the
        provided key.
        """
        def handler(event: wx.EVT_BUTTON):
            # Change dialog type depending on whether parameter requires a
            # directory or file
            if directory:
                file_dialog = wx.DirDialog(
                    self, 'Select a path', style=wx.FD_OPEN)
            else:
                file_dialog = wx.FileDialog(
                    self, 'Select a file', style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                new_path = None
            else:
                new_path = str(file_dialog.GetPath())
                if directory:
                    # Add operating system separator character to end of
                    # directory path
                    new_path = new_path + sep
                log.debug(new_path)

            if new_path:
                for item_key, field in self.controls.items():
                    if item_key == key:
                        field[0].SetValue(new_path)
                parameter_key = self.params[key]
                self.params[key] = parameter_key._replace(value=new_path)
                log.debug(f'Selected path: {new_path}')

        return handler


class MainPanel(scrolled.ScrolledPanel):
    """Panel which contains the Form. Responsible for selecting the json data
     to load and handling scrolling."""

    def __init__(self, parent, title='BCI Parameters',
                 json_file='bcipy/parameters/parameters.json'):
        super(MainPanel, self).__init__(parent, -1)
        self.json_file = json_file
        vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox = vbox
        self.params = {}

        loading_box = wx.BoxSizer(wx.VERTICAL)

        self.loaded_from = static_text_control(
            self,
            label=f'Loaded from: {json_file}',
            size=14)
        loading_box.Add(self.loaded_from)
        loading_box.AddSpacer(10)

        self.get_parameters_from_file(self.json_file)

        self.form = Form(self, json_file)
        vbox.Add(static_text_control(self, label=title, size=20),
                 0, wx.TOP | wx.ALIGN_CENTER_HORIZONTAL, border=5)
        vbox.AddSpacer(10)

        save_box = wx.BoxSizer(wx.HORIZONTAL)

        self.loadButton = wx.Button(self, label='Load')
        self.loadButton.Bind(wx.EVT_BUTTON, self.onLoad)
        save_box.Add(self.loadButton)
        save_box.AddSpacer(10)
        self.saveButton = wx.Button(self, label='Save')
        self.saveButton.Bind(wx.EVT_BUTTON, self.onSave)
        save_box.Add(self.saveButton)
        save_box.AddSpacer(10)
        self.restoreDefaultsButton = wx.Button(self, label='Restore Defaults')
        self.restoreDefaultsButton.Bind(wx.EVT_BUTTON, self.restoreDefaults)
        save_box.Add(self.restoreDefaultsButton)

        loading_box.Add(save_box)
        loading_box.AddSpacer(10)
        loading_box.Add(static_text_control(self, label='BciPy version {}'.format(bcipy_version()), size=14))

        # Used for displaying help messages to the user.
        self.flash_msg = static_text_control(self, label='', size=14,
                                             color='FOREST GREEN')
        loading_box.Add(self.flash_msg)

        vbox.Add(loading_box, 0, wx.ALL, border=10)
        vbox.AddSpacer(10)
        vbox.Add(self.form)
        self.SetSizer(vbox)
        self.SetupScrolling()

    def get_parameters_from_file(self, file_name: str):
        self.loaded_from.SetLabel(f'Loaded from: {file_name}')

        with open(file_name) as f:
            data = f.read()

        config = json.loads(data)
        self.params = {k: Parameter(*v.values()) for k, v in config.items()}

    def onLoad(self, event: wx.EVT_BUTTON) -> None:
        """Event handler to load the form data from a different json file."""
        log.debug('Loading parameters file')

        with wx.FileDialog(self, 'Open parameters file',
                           wildcard='JSON files (*.json)|*.json',
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fd:
            if fd.ShowModal() == wx.ID_CANCEL:
                return     # the user changed their mind
            load_file = fd.GetPath()

            self.get_parameters_from_file(load_file)

            self.params, missing_keys = check_if_parameters_contains_all_keys(self.params, load_file)

            if missing_keys:
                for each_key in missing_keys:
                    self.params[each_key] = Parameter(*self.params[each_key].values())

                dialog = wx.MessageDialog(
                    self, "Parameters file {} is missing keys {}. The default "
                    "values for these keys will be loaded.".format(
                        load_file, str(missing_keys)), 'Warning', wx.OK |
                    wx.ICON_EXCLAMATION)
                dialog.ShowModal()
                dialog.Destroy()

        self.write_parameters_location_txt(load_file)

        self.refresh_form(load_file)

    def restoreDefaults(self, event: wx.EVT_BUTTON) -> None:
        self.get_parameters_from_file(DEFAULT_PARAMETERS_LOCATION)
        self.write_parameters_location_txt(DEFAULT_PARAMETERS_LOCATION)
        self.refresh_form(DEFAULT_PARAMETERS_LOCATION)

    def onSave(self, event: wx.EVT_BUTTON) -> None:
        with wx.FileDialog(self, 'Save parameters file',
                           wildcard='JSON files (*.json)|*.json',
                           style=wx.FD_SAVE) as fd:
            if fd.ShowModal() == wx.ID_CANCEL:
                return     # the user changed their mind
            save_file = fd.GetPath()

        if path.samefile('bcipy/parameters/parameters.json', save_file):
            dialog = wx.MessageDialog(
                self, "This will overwrite the default parameters.json. Your changes "
                "will be overwritten when BciPy is upgraded.", 'Warning',
                wx.CANCEL | wx.OK | wx.ICON_EXCLAMATION)
            if dialog.ShowModal() == wx.ID_CANCEL:
                return
            dialog.Destroy()

        self.loaded_from.SetLabel(f'Loaded from: {save_file}')
        self.json_file = save_file
        self.params['parameter_location'] = self.params['parameter_location']._replace(value=save_file)

        log.debug('Saving parameter data')

        with open(save_file, 'w') as outfile:
            json.dump({k: v._asdict() for k, v in self.params.items()},
                      outfile, indent=JSON_INDENT)

        self.write_parameters_location_txt(save_file)

        self.refresh_form()

        dialog = wx.MessageDialog(
            self, "Parameters Successfully Updated!", 'Info',
            wx.OK | wx.ICON_INFORMATION)
        dialog.ShowModal()
        dialog.Destroy()

    def refresh_form(self, load_file=None):
        self.vbox.Hide(self.form)
        self.form = Form(self, json_file=self.json_file, load_file=load_file)
        self.vbox.Add(self.form)
        self.SetupScrolling()

    def write_parameters_location_txt(self, location):
        with open('parameters_location.txt', 'w') as outfile:
            outfile.write(location)

    def OnChildFocus(self, event):
        event.Skip()


def main(title='BCI Parameters', size=(650, 550),
         json_file='bcipy/parameters/parameters.json'):
    """Set up the GUI components and start the main loop."""

    app = wx.App(0)
    frame = wx.Frame(None, wx.ID_ANY, size=size, title=title)
    fa = MainPanel(frame, title=title, json_file=json_file)
    frame.Show()
    app.MainLoop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Command line utility for adding arguments/ paths via command line
    parser.add_argument('-p', '--parameters', default='bcipy/parameters/parameters.json',
                        help='Parameter location. Must be in parameters directory. Pass as parameters/parameters.json')
    args = parser.parse_args()
    main(json_file=args.parameters)
