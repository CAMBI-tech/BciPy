"""BCIUI module.

This module provides the base UI components and utilities for building BciPy's
graphical user interfaces. It includes base classes for UI elements, dynamic
lists, and utility functions for common UI operations.
"""

import sys
from typing import Callable, List, Optional, Type, Any, Dict

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLayout, QMessageBox,
                             QPushButton, QScrollArea, QSizePolicy,
                             QVBoxLayout, QWidget)

from bcipy.config import BCIPY_ROOT


class BCIUI(QWidget):
    """Base class for BciPy user interfaces.

    This class provides common functionality for all BciPy UI components,
    including layout management, styling, and utility methods.

    Attributes:
        contents (QVBoxLayout): Main vertical layout container.
        center_content_vertically (bool): Whether to center content vertically.
    """

    contents: QVBoxLayout
    center_content_vertically: bool = False

    def __init__(self, title: str = "BCIUI", default_width: int = 500, default_height: int = 600) -> None:
        """Initialize the BCIUI base class.

        Args:
            title (str): Window title. Defaults to "BCIUI".
            default_width (int): Default window width. Defaults to 500.
            default_height (int): Default window height. Defaults to 600.
        """
        super().__init__()
        self.resize(default_width, default_height)
        self.setWindowTitle(title)
        self.contents = QVBoxLayout()
        self.setLayout(self.contents)

    def app(self) -> None:
        """Initialize the application UI.

        This method should be overridden by subclasses to set up their specific UI elements.
        """
        ...

    def apply_stylesheet(self) -> None:
        """Apply the BciPy stylesheet to the UI.

        Loads and applies the CSS stylesheet from the BciPy configuration.
        """
        stylesheet_path = f'{BCIPY_ROOT}/gui/bcipy_stylesheet.css'  # TODO: move to config
        with open(stylesheet_path, "r") as f:
            stylesheet = f.read()
        self.setStyleSheet(stylesheet)

    def display(self) -> None:
        """Display the UI window and apply the stylesheet.

        Initializes the UI, applies vertical centering if configured,
        and shows the window.
        """
        self.app()
        if not self.center_content_vertically:
            self.contents.addStretch()
        self.apply_stylesheet()
        self.show()

    def show_alert(self, alert_text: str) -> int:
        """Show an alert dialog with the specified text.

        Args:
            alert_text (str): Text to display in the alert dialog.

        Returns:
            int: The result code from the message box.
        """
        msg = QMessageBox()
        msg.setText(alert_text)
        msg.setWindowTitle("Alert")
        return msg.exec()

    @staticmethod
    def centered(widget: QWidget) -> QHBoxLayout:
        """Create a centered horizontal layout for a widget.

        Args:
            widget (QWidget): Widget to center.

        Returns:
            QHBoxLayout: Layout with the widget centered horizontally.
        """
        layout = QHBoxLayout()
        layout.addStretch()
        layout.addWidget(widget)
        layout.addStretch()
        return layout

    @staticmethod
    def make_list_scroll_area(widget: QWidget) -> QScrollArea:
        """Create a scrollable area for a widget.

        Args:
            widget (QWidget): Widget to make scrollable.

        Returns:
            QScrollArea: Scrollable area containing the widget.
        """
        scroll_area = QScrollArea()
        scroll_area.setWidget(widget)
        scroll_area.setWidgetResizable(True)
        return scroll_area

    @staticmethod
    def make_toggle(
        on_button: QPushButton,
        off_button: QPushButton,
        on_action: Callable = lambda: None,
        off_action: Callable = lambda: None,
    ) -> None:
        """Connect two buttons to toggle between each other and call passed methods.

        Args:
            on_button (QPushButton): Button to toggle on.
            off_button (QPushButton): Button to toggle off.
            on_action Callable: Function to call when on_button is clicked.
            off_action Callable: Function to call when off_button is clicked.
        """
        off_button.hide()

        def toggle_off() -> None:
            on_button.hide()
            off_button.show()
            off_action()

        def toggle_on() -> None:
            on_button.show()
            off_button.hide()
            on_action()

        on_button.clicked.connect(toggle_off)
        off_button.clicked.connect(toggle_on)

    def hide(self) -> None:
        """Hide the UI window."""
        self.hide()


class SmallButton(QPushButton):
    """A small button with a fixed size.

    This button is styled with a specific CSS class and fixed size policy.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the small button.

        Args:
            *args: Positional arguments passed to QPushButton.
            **kwargs: Keyword arguments passed to QPushButton.
        """
        super().__init__(*args, **kwargs)
        self.setProperty("class", "small-button")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


class DynamicItem(QWidget):
    """A widget that can be dynamically added and removed from the UI.

    This widget emits a signal when removed and can store arbitrary data.

    Attributes:
        on_remove (pyqtSignal): Signal emitted when the item is removed.
        data (Dict[str, Any]): Dictionary for storing arbitrary data.
    """

    on_remove: pyqtSignal = pyqtSignal()
    data: Dict[str, Any] = {}

    def remove(self) -> None:
        """Remove the widget from its parent DynamicList.

        Emits the on_remove signal and triggers widget deletion.
        """
        self.on_remove.emit()


class DynamicList(QWidget):
    """A list of QWidgets that can be dynamically updated.

    This widget manages a list of DynamicItems that can be added, removed,
    and reordered.

    Attributes:
        widgets (List[QWidget]): List of managed widgets.
    """

    widgets: List[QWidget]

    def __init__(self, layout: Optional[QLayout] = None) -> None:
        """Initialize the dynamic list.

        Args:
            layout (Optional[QLayout]): Layout to use. Defaults to QVBoxLayout.
        """
        super().__init__()
        if layout is None:
            layout = QVBoxLayout()
        self.setLayout(layout)
        self.widgets = []

    def __len__(self) -> int:
        """Get the number of widgets in the list.

        Returns:
            int: Number of widgets.
        """
        return len(self.widgets)

    def add_item(self, item: DynamicItem) -> None:
        """Add a DynamicItem to the list.

        Args:
            item (DynamicItem): Item to add to the list.
        """
        self.widgets.append(item)
        item.on_remove.connect(lambda: self.remove_item(item))
        layout = self.layout()
        if layout:
            layout.addWidget(item)

    def move_item(self, item: DynamicItem, new_index: int) -> None:
        """Move a DynamicItem to a new index in the list.

        Args:
            item (DynamicItem): Item to move.
            new_index (int): New index for the item.

        Raises:
            IndexError: If new_index is out of range.
        """
        if new_index < 0 or new_index >= len(self):
            raise IndexError(f"Index out of range for length {len(self)}")

        self.widgets.pop(self.widgets.index(item))
        self.widgets.insert(new_index, item)
        layout = self.layout()
        if layout:
            layout.removeWidget(item)
            layout.insertWidget(new_index, item)

    def index(self, item: DynamicItem) -> int:
        """Get the index of a DynamicItem in the list.

        Args:
            item (DynamicItem): Item to find the index of.

        Returns:
            int: Index of the item in the list.
        """
        return self.widgets.index(item)

    def remove_item(self, item: DynamicItem) -> None:
        """Remove a DynamicItem from the list.

        Args:
            item (DynamicItem): Item to remove.
        """
        self.widgets.remove(item)
        layout = self.layout()
        if layout:
            layout.removeWidget(item)
        item.deleteLater()

    def clear(self) -> None:
        """Remove all items from the list."""
        for widget in self.widgets:
            layout = self.layout()
            if layout:
                layout.removeWidget(widget)
            widget.deleteLater()
        self.widgets = []

    def list(self) -> List[Dict[str, Any]]:
        """Get a list of data dictionaries from all items.

        Returns:
            List[Dict[str, Any]]: List of data dictionaries.
        """
        return [widget.data for widget in self.widgets]

    def list_property(self, prop: str) -> List[Any]:
        """Get a list of values for a given property of each DynamicItem's data dictionary.

        Args:
            prop (str): Property name to get values for.

        Returns:
            List[Any]: List of values for the given property.
        """
        return [widget.data[prop] for widget in self.widgets]


def run_bciui(ui: Type[BCIUI], *args: Any, **kwargs: Any) -> int:
    """Run a BCIUI instance.

    Args:
        ui (Type[BCIUI]): BCIUI class to instantiate.
        *args: Positional arguments for the UI class.
        **kwargs: Keyword arguments for the UI class.

    Returns:
        int: Application exit code.
    """
    app = QApplication(sys.argv).instance()
    if not app:
        app = QApplication(sys.argv)
    ui_instance = ui(*args, **kwargs)
    ui_instance.display()
    return app.exec()
