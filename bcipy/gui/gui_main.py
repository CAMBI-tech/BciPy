import os
import sys
import json
from time import localtime, strftime
import logging

from enum import Enum

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QAction,
    QLabel,
    QLineEdit,
    QComboBox,
    QMessageBox,
    QGridLayout,
    QSpinBox,
    QDoubleSpinBox,
    QMessageBox)
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import pyqtSlot

class AlertMessageType(Enum):
    WARN = QMessageBox.Warning
    QUESTION = QMessageBox.Question
    INFO = QMessageBox.Information
    CRIT = QMessageBox.Critical


class AlertResponse(Enum):
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


class BCIGui(QtWidgets.QMainWindow):
    """GUI for BCIGui."""

    title = 'BciPy Main GUI'

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

    def show_gui(self):
        """Show GUI."""
        self.build_assets()
        self.init_ui()

    def close_gui(self):
        pass

    def build_assets(self):
        pass
    
    def init_ui(self):
        """Init UI."""
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
    def default_on_clicked(self):
        self.logger.debug('Button clicked!')

    @pyqtSlot()
    def default_on_dropdown(self):
        """on_dropdown.

        Default event to bind to comboboxes
        """
        self.logger.debug('Dropdown selected!')

    @pyqtSlot()
    def buttonClicked(self):
        sender = self.sender()
        self.logger.debug(sender.text() + ' was pressed')
        self.logger.debug(sender.get_id())
    
    def add_button(self, message: str, position: list, size: list, id=-1,
                   background_color: str = 'white',
                   text_color: str = 'default',
                   button_type: str = None,
                   action = None) -> PushButton:
        """Add Button.
        
        Size : width and height in pixels
        """
        btn = PushButton(message, self.window)
        btn.id = id
        btn.move(position[0], position[1])
        btn.resize(size[0], size[1])

        btn.setStyleSheet(f'background-color: {background_color}; color: {text_color};')


        if action:
            btn.clicked.connect(action)
        else:
            btn.clicked.connect(self.buttonClicked)

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

        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(message_type.value)

        if okay_to_exit:
            msg.setStandardButtons(AlertResponse.OK.value)
        elif okay_or_cancel:
            msg.setStandardButtons(AlertResponse.OK.value | AlertResponse.CANCEL.value)

        return msg.exec_()

def app(args):
    """Main app registry.

    Passes args from main and initializes the app
    """
    return QApplication(args)


def start_app():
    """Start BCIGui."""
    bcipy_gui = app(sys.argv)
    ex = BCIGui(title='BCI GUI', height=650, width=650, background_color='white')

    ex.throw_alert_message('title', 'test', okay_to_exit=True)
    ex.add_button(message='Test Button', position=[200, 300], size=[100, 100], id=1)
    # ex.add_image(path='../static/images/gui_images/bci_cas_logo.png', position=[50, 50], size=200)
    # ex.add_static_textbox(text='Test static text', background_color='black', text_color='white', position=[100, 20], wrap_text=True)
    # ex.add_combobox(position=[100, 100], size=[100, 100], items=['first', 'second', 'third'], editable=True)
    # ex.add_text_input(position=[100, 100], size=[100, 100])
    ex.show_gui()

    sys.exit(bcipy_gui.exec_())

if __name__ == '__main__':
    start_app()