# pylint: disable=E0611
import logging
import os
import sys
from enum import Enum
from typing import List

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                             QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                             QMainWindow, QMessageBox, QPushButton,
                             QVBoxLayout, QWidget)

from bcipy.helpers.parameters import Parameter


def font(size: int = 14, font_family: str = 'Helvetica') -> QFont:
    """Create a Font object with the given parameters."""
    return QFont(font_family, size, QFont.Normal)


def static_text_control(parent,
                        label: str,
                        color: str = 'black',
                        size: int = 14,
                        font_family: str = 'Helvetica') -> QLabel:
    """Creates a static text control with the given font parameters. Useful for
    creating labels and help components."""
    static_text = QLabel(parent)
    static_text.setWordWrap(True)
    static_text.setText(label)
    static_text.setStyleSheet(f'color: {color};')
    static_text.setFont(font(size, font_family))
    return static_text


class AlertMessageType(Enum):
    """Alert Message Type.

    Custom enum used to abstract PyQT message types from downstream users.
    """
    WARN = QMessageBox.Warning
    QUESTION = QMessageBox.Question
    INFO = QMessageBox.Information
    CRIT = QMessageBox.Critical


class AlertResponse(Enum):
    """Alert Response.

    Custom enum used to abstract PyQT alert responses from downstream users.
    """
    OK = QMessageBox.Ok
    CANCEL = QMessageBox.Cancel
    YES = QMessageBox.Yes
    NO = QMessageBox.No


class PushButton(QPushButton):
    """PushButton.

    Custom Button to store unique identifiers which are required for coordinating
    events across multiple buttons."""
    id = None

    def get_id(self):
        if not self.id:
            raise Exception('No ID set on PushButton')

        return self.id

class ComboBox(QComboBox):
    """ComboBox with the same interface as a QLineEdit for getting and setting values.

    Parameters:
    -----------
      options - list of items to appear in the lookup
      selected_value - selected item; if this is not one of the options and new option will be
        created for this value.
    """

    def __init__(self, options: List[str], selected_value: str, **kwargs):
        super(ComboBox, self).__init__(**kwargs)
        self.options = options
        self.addItems(self.options)
        self.setText(selected_value)
        self.setEditable(True)

    def setText(self, value: str):
        """Sets the current index to the given value. If the value is not in the list of
        options it will be added."""
        if value not in self.options:
            self.options = [value] + self.options
            self.clear()
            self.addItems(self.options)
        self.setCurrentIndex(self.options.index(value))

    def text(self):
        """Gets the currentText."""
        return self.currentText()


class FormInput(QWidget):
    """A form element with a label, help tip, and input control. The default control is a text
    input. This object may be subclassed to specialize the input control or the arrangement of
    widgets. FormInputs have the ability to show and hide themselves.

    Parameters:
    ----------
      parameter - the option which will be configured using this input. Used to populate the label,
        help text, and relevant options.
      help_font_size - font size for the help text.
      help_color - color of the help text.
    """

    def __init__(self,
                 parameter: Parameter,
                 help_font_size: int = 12,
                 help_color: str = 'darkgray'):
        super(FormInput, self).__init__()
        self.parameter = parameter

        self.label = self.init_label()
        self.help_tip = self.init_help(help_font_size, help_color)
        self.control = self.init_control()

        self.init_layout()

    def init_label(self) -> QWidget:
        """Initize the label widget."""
        return static_text_control(None, label=self.parameter.readableName)

    def init_help(self, font_size: int, color: str) -> QWidget:
        """Initialize the help text widget."""
        if self.parameter.readableName == self.parameter.helpTip:
            return None
        return static_text_control(None,
                                   label=self.parameter.helpTip,
                                   size=font_size,
                                   color=color)

    def init_control(self) -> QWidget:
        """Initialize the form control widget."""
        # Default is a text input
        return QLineEdit(self.parameter.value)

    def init_layout(self):
        """Initialize the layout by adding the label, help, and control widgets."""
        self.vbox = QVBoxLayout()
        if self.label:
            self.vbox.addWidget(self.label)
        if self.help_tip:
            self.vbox.addWidget(self.help_tip)
        self.vbox.addWidget(self.control)
        self.setLayout(self.vbox)

    def value(self) -> str:
        """Returns the value associated with the form input."""
        if self.control:
            return self.control.text()
        return None

    def matches(self, term: str) -> bool:
        """Returns True if the input matches the given text, otherwise False."""
        text = term.lower()
        return (text in self.parameter.readableName.lower()) or (
            self.parameter.helpTip and text in self.parameter.helpTip.lower()
        ) or text in self.value().lower()

    def show(self):
        """Show this widget, and all child widgets."""
        for widget in self.widgets():
            if widget:
                widget.setVisible(True)

    def hide(self):
        """Hide this widget, and all child widgets."""
        for widget in self.widgets():
            if widget:
                widget.setVisible(False)

    def widgets(self) -> List[QWidget]:
        """Returns a list of self and child widgets. List may contain None values."""
        return [self.label, self.help_tip, self.control, self]


class BoolInput(FormInput):
    """Checkbox form input used for boolean configuration parameters. Overrides FormInput to provide
    a CheckBox control. Help text is not displayed for checkbox items."""

    def __init__(self, parameter: Parameter, **kwargs):
        super(BoolInput, self).__init__(parameter, **kwargs)

    def init_label(self) -> QWidget:
        """Override. Checkboxes do not have a separate label."""
        return None

    def init_help(self, font_size: int, color: str) -> QWidget:
        """Override. Checkboxes do not display help."""
        return None

    def init_control(self):
        """Override to create a checkbox."""
        ctl = QCheckBox(self.parameter.readableName)
        ctl.setChecked(self.parameter.value == 'true')
        ctl.setFont(font())
        return ctl

    def value(self) -> str:
        return 'true' if self.control.isChecked() else 'false'


class SelectionInput(FormInput):
    """Input to select from a list of options."""

    def __init__(self, parameter: Parameter, **kwargs):
        super(SelectionInput, self).__init__(parameter, **kwargs)

    def init_control(self) -> QWidget:
        """Override to create a Combobox."""
        return ComboBox(self.parameter.recommended_values,
                        self.parameter.value)


class TextInput(FormInput):
    """Text field input."""

    def __init__(self, parameter: Parameter, **kwargs):
        super(TextInput, self).__init__(parameter, **kwargs)


class FileInput(FormInput):
    """Input for selecting a file or directory."""

    def __init__(self, parameter: Parameter, **kwargs):
        super(FileInput, self).__init__(parameter, **kwargs)

    def init_control(self) -> QWidget:
        """Override to create either a selection list or text field depending
        on whether there are recommended values."""
        param = self.parameter
        if isinstance(self.parameter.recommended_values, list):
            return ComboBox(param.recommended_values, param.value)
        return QLineEdit(param.value)

    def init_button(self) -> QWidget:
        """Creates a Button to initiate the file/directory dialog."""
        btn = QPushButton('...')
        btn.setFixedWidth(40)
        btn.clicked.connect(self.prompt_path)
        return btn

    def prompt_path(self):
        """Prompts the user with a FileDialog. Updates the control's value with the file or
        directory if one was selected."""
        if self.parameter.type == 'directorypath':
            name = QFileDialog.getExistingDirectory(caption='Select a path')
        else:
            name, _ = QFileDialog.getOpenFileName(caption='Select a file',
                                                  filter='All Files (*)')

        if name:
            self.control.setText(name)

    def init_layout(self):
        """Overrides the layout to add the file dialog button."""
        self.button = self.init_button()
        self.vbox = QVBoxLayout()
        if self.label:
            self.vbox.addWidget(self.label)
        if self.help_tip:
            self.vbox.addWidget(self.help_tip)
        hbox = QHBoxLayout()
        hbox.addWidget(self.control)
        hbox.addWidget(self.button)
        self.vbox.addLayout(hbox)
        self.setLayout(self.vbox)

    def widgets(self) -> List[QWidget]:
        """Override to include button."""
        return super().widgets() + [self.button]

def parameter_input(param: Parameter) -> FormInput:
    """Construct a FormInput for the given parameter based on its python type and other
    attributes."""

    if param.type == 'bool':
        return BoolInput(param)
    if 'path' in param.type:
        return FileInput(param)
    if isinstance(param.recommended_values, list):
        return SelectionInput(param)
    return TextInput(param)

class SearchInput(QWidget):
    """Search input widget. Consists of a text input field and a Clear button.
    Text changes to the input are passed to the on_search action.
    The cancel button clears the input.

    Parameters:
    -----------
        on_search - search function to call. Takes a single str parameter, the
            contents of the text box.
    """

    def __init__(self, on_search, font_size: int = 10):
        super(SearchInput, self).__init__()

        self.on_search = on_search

        self.hbox = QHBoxLayout()
        self.hbox.addWidget(static_text_control(None, label='Search: '))

        self.input = QLineEdit()
        self.input.setStyleSheet(
            f"font-size: {font_size}px; line-height: {font_size}px")
        self.input.textChanged.connect(self._search)
        self.hbox.addWidget(self.input)

        cancel_btn = QPushButton('Clear')
        cancel_btn.setStyleSheet(f"font-size: {font_size}px;")
        cancel_btn.clicked.connect(self._cancel)
        self.hbox.addWidget(cancel_btn)

        self.setLayout(self.hbox)

    def _search(self):
        self.on_search(self.input.text())

    def _cancel(self):
        self.input.clear()
        self.input.repaint()


class BCIGui(QMainWindow):
    """GUI for BCIGui."""

    def __init__(self, title: str, width: int, height: int, background_color: str):
        super(BCIGui, self).__init__()
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(name)s - %(levelname)s - %(message)s')
        self.logger = logging

        self.window = QWidget()

        self.buttons = []
        self.input_text = []
        self.static_text = []
        self.images = []
        self.comboboxes = []

        # set window properties
        self.window.setStyleSheet(f'background-color: {background_color};')

        # determines where on the screen the gui first appears
        self.x = 500
        self.y = 250

        self.title = title

        # determines height/width of window
        self.width = width
        self.height = height

    def show_gui(self) -> None:
        """Show GUI.

        Build all registered assets and initialize the interface.
        """
        self.build_assets()
        self.init_ui()

    def build_buttons(self) -> None:
        """Build Buttons."""
        ...

    def build_images(self) -> None:
        """Build Images."""
        ...

    def build_text(self) -> None:
        """Build Text."""
        ...

    def build_inputs(self) -> None:
        """Build Inputs."""
        ...

    def build_assets(self) -> None:
        """Build Assets.

        Build add registered asset types.
        """
        self.build_text()
        self.build_inputs()
        self.build_buttons()
        self.build_images()

    def init_ui(self) -> None:
        """Initalize UI.

        This method sets up the grid and window. Finally, it calls the show method to make
            the window visible.
        """
        # create a grid system to keep everythin aligned
        self.grid = QGridLayout()
        self.grid.setSpacing(2)
        self.window.setLayout(self.grid)

        # main window setup
        self.window.setWindowTitle(self.title)
        self.window.setGeometry(self.x, self.y, self.width, self.height)

        # show the window
        self.window.show()

    @pyqtSlot()
    def default_on_dropdown(self) -> None:
        """Default on dropdown.

        Default event to bind to comboboxes
        """
        self.logger.debug('Dropdown selected!')

    @pyqtSlot()
    def default_button_clicked(self) -> None:
        """Default button clikced.

        The default action for buttons if none are registed.
        """
        sender = self.sender()
        self.logger.debug(sender.text() + ' was pressed')
        self.logger.debug(sender.get_id())

    def add_button(self, message: str, position: list, size: list, id=-1,
                   background_color: str = 'white',
                   text_color: str = 'default',
                   button_type: str = None,
                   action=None) -> PushButton:
        """Add Button."""
        btn = PushButton(message, self.window)
        btn.id = id
        btn.move(position[0], position[1])
        btn.resize(size[0], size[1])

        btn.setStyleSheet(f'background-color: {background_color}; color: {text_color};')

        if action:
            btn.clicked.connect(action)
        else:
            btn.clicked.connect(self.default_button_clicked)

        self.buttons.append(btn)
        return btn

    def add_combobox(self, position: list, size: list, items: list, background_color='default',
                     text_color='default',
                     action=None, editable=False) -> QComboBox:
        """Add combobox."""

        combobox = QComboBox(self.window)
        combobox.move(position[0], position[1])
        combobox.resize(size[0], size[1])

        if action:
            combobox.currentTextChanged.connect(action)
        else:
            combobox.currentTextChanged.connect(self.default_on_dropdown)

        combobox.setStyleSheet(f'background-color: {background_color}; color: {text_color};')

        if editable:
            combobox.setEditable(True)
        combobox.addItems(items)

        self.comboboxes.append(combobox)
        return combobox

    def add_image(self, path: str, position: list, size: int) -> QLabel:
        """Add Image."""
        if os.path.isfile(path):
            labelImage = QLabel(self.window)
            pixmap = QPixmap(path)
            # ensures the new label size will scale the image itself
            labelImage.setScaledContents(True)
            labelImage.setPixmap(pixmap)
            width = pixmap.width()
            height = pixmap.height()

            if width > height:
                new_width = size
                new_height = size * height / width
            else:
                new_height = size
                new_width = size * width / height

            labelImage.resize(new_width, new_height)
            labelImage.move(position[0], position[1])

            self.images.append(labelImage)
            return labelImage
        raise Exception('Invalid path to image provided')

    def add_static_textbox(self, text: str, position: list,
                           background_color: str = 'white',
                           text_color: str = 'default',
                           size: list = None,
                           font_family="Times",
                           font_size=12,
                           wrap_text=False) -> QLabel:
        """Add Static Text."""

        static_text = QLabel(self.window)
        static_text.setText(text)
        if wrap_text:
            static_text.setWordWrap(True)
        static_text.setStyleSheet(f'background-color: {background_color}; color: {text_color};')
        static_text.move(position[0], position[1])

        text_settings = QFont(font_family, font_size)
        static_text.setFont(text_settings)
        if size:
            static_text.resize(size[0], size[1])

        self.static_text.append(static_text)
        return static_text

    def add_text_input(self, position: list, size: list) -> QLineEdit:
        """Add Text Input."""
        textbox = QLineEdit(self.window)
        textbox.move(position[0], position[1])
        textbox.resize(size[0], size[1])

        self.input_text.append(textbox)
        return textbox

    def throw_alert_message(self,
                            title: str,
                            message: str,
                            message_type: AlertMessageType = AlertMessageType.INFO,
                            okay_to_exit: bool = False,
                            okay_or_cancel: bool = False) -> QMessageBox:
        """Throw Alert Message."""

        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(message_type.value)

        if okay_to_exit:
            msg.setStandardButtons(AlertResponse.OK.value)
        elif okay_or_cancel:
            msg.setStandardButtons(AlertResponse.OK.value | AlertResponse.CANCEL.value)

        return msg.exec_()

    def get_filename_dialog(
            self,
            message: str = 'Open File',
            file_type: str = 'All Files (*)',
            location: str = "") -> str:
        """Get Filename Dialog."""
        file_name, _ = QFileDialog.getOpenFileName(self.window, message, location, file_type)
        return file_name


def app(args) -> QApplication:
    """Main app registry.

    Passes args from main and initializes the app
    """
    return QApplication(args)


def start_app() -> None:
    """Start BCIGui."""
    bcipy_gui = app(sys.argv)
    ex = BCIGui(title='BCI GUI', height=650, width=650, background_color='white')

    # ex.throw_alert_message(title='title', message='test', okay_to_exit=True)
    ex.get_filename_dialog()
    ex.add_button(message='Test Button', position=[200, 300], size=[100, 100], id=1)
    # ex.add_image(path='../static/images/gui_images/bci_cas_logo.png', position=[50, 50], size=200)
    # ex.add_static_textbox(
    #   text='Test static text',
    #   background_color='black',
    #   text_color='white',
    #   position=[100, 20],
    #   wrap_text=True)
    # ex.add_combobox(position=[100, 100], size=[100, 100], items=['first', 'second', 'third'], editable=True)
    # ex.add_text_input(position=[100, 100], size=[100, 100])
    ex.show_gui()

    sys.exit(bcipy_gui.exec_())


if __name__ == '__main__':
    start_app()
