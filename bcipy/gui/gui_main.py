import os
import sys
import json
from time import localtime, strftime
import logging

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QAction,
    QLabel,
    QLineEdit,
    QMessageBox,
    QGridLayout,
    QSpinBox,
    QDoubleSpinBox,
    QMessageBox)
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSlot


class BCIGui(QtWidgets.QMainWindow):
    """GUI for BCIGui."""

    title = 'BciPy Main GUI'

    def __init__(self, title: str, width: int, height: int, background_color: str):
        super(BCIGui, self).__init__()

        
        self.window = QWidget()
        self.buttons = []
        self.input_text = []
        self.static_text = []
        self.images = []
        self.comboboxes = []

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(name)s - %(levelname)s - %(message)s')
        self.logger = logging

        # determines where on the screen the gui first appears
        self.x = 500
        self.y = 250

        self.title = title

        # determines height/width of window
        self.width = width
        self.height = height

    def show_gui(self):
        """Show GUI."""
        self.init_ui()

    def close_gui(self):
        pass

    def init_ui(self):
        """Init UI."""
        # create a grid system to keep everythin aligned
        self.grid = QGridLayout()
        self.grid.setSpacing(2)

        # main window setup
        self.window.setWindowTitle(self.title)
        self.window.setGeometry(self.x, self.y, self.width, self.height)

        # inputs and buttons
        # self.create_inputs((100, 50), 'button')
        self.window.setLayout(self.grid)

        # show the window
        self.window.show()

    @pyqtSlot()
    def default_on_clicked(self):
        print('clicked!')
    
    def add_button(self, message: str, position: list, size: list,
                   button_type: str = None, color: str = None,
                   action = default_on_clicked, id=-1) -> None:
        """Add Button.
        
        Size : width and height in pixels
        """
        btn = QPushButton(message, self.window)
        btn.move(position[0], position[1])
        btn.resize(size[0], size[1])
        btn.clicked.connect(action)

        self.buttons.append(btn)

    def add_image(self, path: str, position: list, size: int) -> None:
        """Add Image."""
        if os.path.isfile(path):
            labelImage = QLabel(self.window)
            pixmap = QPixmap(path)
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
        else:
            raise Exception('Invalid path to image provided')

    def add_static_text(self, text: str, position: str,
                        color: str, size: int,
                        font_family=None) -> None:
        """Add Text."""

        static_text = wx.StaticText(
            self.panel, pos=position,
            label=text)
        static_text.SetForegroundColour(color)
        font = wx.Font(
            size,
            font_family,
            wx.FONTSTYLE_NORMAL,
            wx.FONTWEIGHT_LIGHT)
        static_text.SetFont(font)

        self.static_text.append(static_text)

def app(args):
    """Main app registry.

    Passes args from main and initializes the app
    """
    return QApplication(args)


def start_app():
    """Start BCIGui."""
    bcipy_gui = app(sys.argv)
    ex = BCIGui('title', 400, 400, 'blue')

    ex.add_button(message='test', position=[200, 300], size=[100, 100], id=1)
    ex.add_image(path='../static/images/gui_images/bci_cas_logo.png', position=[50, 50], size=200)
    ex.show_gui()

    sys.exit(bcipy_gui.exec_())

if __name__ == '__main__':
    start_app()