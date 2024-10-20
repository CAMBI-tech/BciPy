from typing import Callable, Type
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QLayout,
    QSizePolicy,
    QMessageBox,
    QApplication,
)
from typing import Optional, List
from bcipy.config import BCIPY_ROOT
import sys


class BCIUI(QWidget):
    contents: QVBoxLayout
    center_content_vertically: bool = False

    def __init__(self, title: str = "BCIUI", default_width: int = 500, default_height: int = 600) -> None:
        super().__init__()
        self.resize(default_width, default_height)
        self.setWindowTitle(title)
        self.contents = QVBoxLayout()
        self.setLayout(self.contents)

    def app(self):
        ...

    def apply_stylesheet(self) -> None:
        stylesheet_path = f'{BCIPY_ROOT}/gui/bcipy_stylesheet.css'  # TODO: move to config
        with open(stylesheet_path, "r") as f:
            stylesheet = f.read()
        self.setStyleSheet(stylesheet)

    def display(self) -> None:
        # Push contents to the top of the window
        """
        Display the UI window and apply the stylesheet.
        """
        self.app()
        if not self.center_content_vertically:
            self.contents.addStretch()
        self.apply_stylesheet()
        self.show()

    def show_alert(self, alert_text: str) -> int:
        """
        Shows an alert dialog with the specified text.

        PARAMETERS
        ----------
        :param: alert_text: string text to display in the alert dialog.
        """
        msg = QMessageBox()
        msg.setText(alert_text)
        msg.setWindowTitle("Alert")
        return msg.exec()

    @staticmethod
    def centered(widget: QWidget) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.addStretch()
        layout.addWidget(widget)
        layout.addStretch()
        return layout

    @staticmethod
    def make_list_scroll_area(widget: QWidget) -> QScrollArea:
        scroll_area = QScrollArea()
        scroll_area.setWidget(widget)
        scroll_area.setWidgetResizable(True)
        return scroll_area

    @staticmethod
    def make_toggle(
        on_button: QPushButton,
        off_button: QPushButton,
        on_action: Optional[Callable] = lambda: None,
        off_action: Optional[Callable] = lambda: None,
    ) -> None:
        """
        Connects two buttons to toggle between eachother and call passed methods

        PARAMETERS
        ----------
        :param: on_button: QPushButton to toggle on
        :param: off_button: QPushButton to toggle off
        :param: on_action: function to call when on_button is clicked
        :param: off_action: function to call when off_button is clicked

        """
        off_button.hide()

        def toggle_off():
            on_button.hide()
            off_button.show()
            off_action()

        def toggle_on():
            on_button.show()
            off_button.hide()
            on_action()

        on_button.clicked.connect(toggle_off)
        off_button.clicked.connect(toggle_on)

    def hide(self) -> None:
        """Close the UI window"""
        self.hide()


class SmallButton(QPushButton):
    """A small button with a fixed size"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setProperty("class", "small-button")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


class DynamicItem(QWidget):
    """A widget that can be dynamically added and removed from the ui"""

    on_remove: pyqtSignal = pyqtSignal()
    data: dict = {}

    def remove(self) -> None:
        """Remove the widget from it's parent DynamicList, removing it from the UI and deleting it"""
        self.on_remove.emit()


class DynamicList(QWidget):
    """A list of QWidgets that can be dynamically updated"""

    widgets: List[QWidget]

    def __init__(self, layout: Optional[QLayout] = None):
        super().__init__()
        if layout is None:
            layout = QVBoxLayout()
        self.setLayout(layout)
        self.widgets = []

    def __len__(self):
        return len(self.widgets)

    def add_item(self, item: DynamicItem) -> None:
        """
        Add a DynamicItem to the list.

        PARAMETERS
        ----------
        :param: item: DynamicItem to add to the list.
        """
        self.widgets.append(item)
        item.on_remove.connect(lambda: self.remove_item(item))
        self.layout().addWidget(item)

    def move_item(self, item: DynamicItem, new_index: int) -> None:
        """
        Move a DynamicItem to a new index in the list.

        PARAMETERS
        ----------
        :param: item: A reference to the DynamicItem in the list to be moved.
        :param: new_index: int new index to move the item to.
        """
        if new_index < 0 or new_index >= len(self):
            raise IndexError(f"Index out of range for length {len(self)}")

        self.widgets.pop(self.widgets.index(item))
        self.widgets.insert(new_index, item)
        self.layout().removeWidget(item)
        self.layout().insertWidget(new_index, item)

    def index(self, item: DynamicItem) -> int:
        """
        Get the index of a DynamicItem in the list.

        PARAMETERS
        ----------
        :param: item: A reference to the DynamicItem in the list to get the index of.

        Returns
        -------
            The index of the item in the list.
        """
        return self.widgets.index(item)

    def remove_item(self, item: DynamicItem) -> None:
        """
        Remove a DynamicItem from the list.

        PARAMETERS
        ----------
        :param: item: A reference to the DynamicItem to remove from the list
        """
        self.widgets.remove(item)
        self.layout().removeWidget(item)
        item.deleteLater()

    def clear(self) -> None:
        """Remove all items from the list"""
        for widget in self.widgets:
            self.layout().removeWidget(widget)
            widget.deleteLater()
        self.widgets = []

    def list(self):
        return [widget.data for widget in self.widgets]

    def list_property(self, prop: str):
        """
        Get a list of values for a given property of each DynamicItem's data dictionary.

        PARAMETERS
        ----------
        :param: prop: string property name to get the values of.

        Returns
        -------
            A list of values for the given property.
        """
        return [widget.data[prop] for widget in self.widgets]


def run_bciui(ui: Type[BCIUI], *args, **kwargs):
    # add app to kwargs
    app = QApplication(sys.argv).instance()
    if not app:
        app = QApplication(sys.argv)
    ui_instance = ui(*args, **kwargs)
    ui_instance.display()
    return app.exec()
