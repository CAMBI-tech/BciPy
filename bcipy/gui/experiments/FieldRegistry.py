import sys
from bcipy.gui.main import BCIGui, app, AlertMessageType, AlertMessageResponse

from bcipy.config import DEFAULT_FIELD_PATH, FIELD_FILENAME
from bcipy.helpers.load import load_fields
from bcipy.helpers.save import save_field_data


class FieldRegistry(BCIGui):
    """Field Registry.

    User interface for creating new fields for use in BCInterface.py.
    """

    padding = 100
    btn_height = 50
    default_text = '...'
    txt_color = 'white'
    input_text_color = 'gray'
    input_color = 'white'
    alert_title = 'Field Registry Alert'
    alert_timeout = 10
    field_types = {
        'Text': 'str',
        'True/False or Yes/No': 'bool',
        'Whole Number': 'int',
        'Decimal Point Number': 'float'
    }

    def __init__(self, *args, **kwargs):
        super(FieldRegistry, self).__init__(*args, **kwargs)

        # Structure of an field:
        #   { name: { help_text: '', type: int} }
        self.update_field_data()

        # These are set in the build_inputs and represent text inputs from the user
        self.name_input = None
        self.helptext_input = None
        self.type_input = None

        self.name = None
        self.helptext = None
        self.type = None

    def build_text(self) -> None:
        """Build Text.

        Build all static text needed for the UI.
        Positions are relative to the height / width of the UI defined in start_app.
        """
        text_x = 25
        text_y = 70
        font_size = 18
        self.add_static_textbox(
            text='Create BciPy Field',
            position=[self.width / 2 - self.padding, 0],
            size=[300, 100],
            background_color=self.background_color,
            text_color=self.txt_color,
            font_size=22
        )
        self.add_static_textbox(
            text='Name',
            position=[text_x, text_y],
            size=[200, 50],
            background_color=self.background_color,
            text_color=self.txt_color,
            font_size=font_size)
        text_y += self.padding
        self.add_static_textbox(
            text='Help Text',
            position=[text_x, text_y],
            size=[300, 50],
            background_color=self.background_color,
            text_color=self.txt_color,
            font_size=font_size)
        text_y += self.padding
        self.add_static_textbox(
            text='Type',
            position=[text_x, text_y],
            size=[300, 50],
            background_color=self.background_color,
            text_color=self.txt_color,
            font_size=font_size)

    def build_inputs(self) -> None:
        """Build Inputs.

        Build all text entry inputs for the UI.
        """
        input_x = 50
        input_y = 120
        input_size = [280, 25]
        self.name_input = self.add_combobox(
            position=[input_x, input_y],
            size=input_size,
            items=[self.default_text],
            editable=True,
            background_color=self.input_color,
            text_color=self.input_text_color)

        input_y += self.padding
        self.helptext_input = self.add_combobox(
            position=[input_x, input_y],
            size=input_size,
            items=[self.default_text],
            editable=True,
            background_color=self.input_color,
            text_color=self.input_text_color)

        input_y += self.padding
        self.type_input = self.add_combobox(
            position=[input_x, input_y],
            size=input_size,
            items=[],
            editable=False,
            background_color=self.input_color,
            text_color=self.input_text_color)

        self.update_type_list()

    def build_buttons(self):
        """Build Buttons.

        Build all buttons necessary for the UI. Define their action on click using the named argument action.
        """
        btn_create_x = self.width - self.padding - 10
        btn_create_y = self.height - 100
        size = 150
        self.add_button(
            message='Create Field', position=[btn_create_x - (size / 2), btn_create_y],
            size=[size, self.btn_height],
            background_color='green',
            action=self.create_field,
            text_color='white')

    def create_field(self) -> None:
        """Create field.

        After inputing all required fields, verified by check_input, add it to the field list and save it.
        """
        if self.check_input():
            self.add_field()
            self.save_fields()
            self.throw_alert_message(
                title=self.alert_title,
                message='Field saved successfully! Please exit window or create another field!',
                message_type=AlertMessageType.INFO,
                message_response=AlertMessageResponse.OTE,
                message_timeout=self.alert_timeout
            )
            self.refresh_inputs()

    def refresh_inputs(self):
        """Refresh Inputs.

        Clear and resets all inputs and related data.
        """
        self.name_input.clear()
        self.name_input.addItem(FieldRegistry.default_text)
        self.update_field_data()
        self.helptext_input.clear()
        self.helptext_input.addItem(FieldRegistry.default_text)
        self.update_type_list()

    def update_field_data(self):
        """Update Field Data.

        Fetches the fields and extracts the registered names.
        """
        self.fields = load_fields()
        self.field_names = list(self.fields.keys())

    def update_type_list(self):
        """Update Type List.

        Update the supported type list, reseting to default_text.
        """
        self.type_input.clear()
        self.type_input.addItem(FieldRegistry.default_text)
        self.type_input.addItems(list(self.field_types.keys()))

    def add_field(self) -> None:
        """Add field:

        Add a new field to the dict of fields. It follows the format:
             { name: { help_text: '', type: int} }
        """
        self.fields[self.name] = {
            'help_text': self.helptext,
            'type': self.field_types[self.type]
        }

    def save_fields(self) -> None:
        """Save field.

        Save the fields registered to the correct path as pulled from system_utils.
        """
        # add fields to the field
        save_field_data(self.fields, DEFAULT_FIELD_PATH, FIELD_FILENAME)

    def build_assets(self) -> None:
        """Build Assets.

        Define the assets to build in the UI.
        """
        self.build_inputs()
        self.build_text()
        self.build_buttons()

    def check_input(self) -> bool:
        """Check Input.

        Checks to make sure user has input all required fields. Currently, only name and summary are required.
        """
        self.name = self.name_input.currentText()
        self.helptext = self.helptext_input.currentText()
        self.type = self.type_input.currentText()

        try:
            if self.name == FieldRegistry.default_text or \
                    self.name == '':
                self.throw_alert_message(
                    title=self.alert_title,
                    message='Please add a field name!',
                    message_type=AlertMessageType.WARN,
                    message_response=AlertMessageResponse.OTE,
                    message_timeout=self.alert_timeout)
                return False
            if self.name in self.field_names:
                self.throw_alert_message(
                    title=self.alert_title,
                    message=(
                        'Field name already registered. \n'
                        'Please use a unique field name! \n'
                        f'Registed names: {self.field_names}'
                    ),
                    message_type=AlertMessageType.WARN,
                    message_response=AlertMessageResponse.OTE,
                    message_timeout=self.alert_timeout)
                return False
            if self.helptext == FieldRegistry.default_text or \
                    self.helptext == '':
                self.throw_alert_message(
                    title=self.alert_title,
                    message='Please add help text to field. \nThis will present when collecting experiment data!',
                    message_type=AlertMessageType.WARN,
                    message_response=AlertMessageResponse.OTE,
                    message_timeout=self.alert_timeout)
                return False
            if self.type == FieldRegistry.default_text:
                self.throw_alert_message(
                    title=self.alert_title,
                    message='Please select a valid field type!',
                    message_type=AlertMessageType.WARN,
                    message_response=AlertMessageResponse.OTE,
                    message_timeout=self.alert_timeout)
                return False
        except Exception as e:
            self.throw_alert_message(
                title=self.alert_title,
                message=f'Error, {e}',
                message_type=AlertMessageType.CRIT,
                message_response=AlertMessageResponse.OTE,
                message_timeout=self.alert_timeout)
            return False
        return True


def start_app() -> None:
    """Start Field Registry."""
    bcipy_gui = app(sys.argv)
    ex = FieldRegistry(
        title='Field Registry',
        height=500,
        width=550,
        background_color='black')

    ex.show_gui()

    sys.exit(bcipy_gui.exec_())


if __name__ == '__main__':
    start_app()
