import os
import sys
import logging

from enum import Enum

from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QFileDialog,
    QWidget,
    QPushButton,
    QLabel,
    QLineEdit,
    QComboBox,
    QMessageBox,
    QGridLayout,
    QMessageBox)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import pyqtSlot

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
                   action = None) -> PushButton:
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
            title:str,
            message:str,
            message_type:AlertMessageType = AlertMessageType.INFO,
            okay_to_exit: bool=False,
            okay_or_cancel: bool=False) -> QMessageBox:
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
            message:str='Open File',
            file_type: str = 'All Files (*)',
            location:str= "") -> str:
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
    # ex.add_static_textbox(text='Test static text', background_color='black', text_color='white', position=[100, 20], wrap_text=True)
    # ex.add_combobox(position=[100, 100], size=[100, 100], items=['first', 'second', 'third'], editable=True)
    # ex.add_text_input(position=[100, 100], size=[100, 100])
    ex.show_gui()

    sys.exit(bcipy_gui.exec_())

if __name__ == '__main__':
    start_app()