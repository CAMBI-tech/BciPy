import sys
import subprocess

from bcipy.gui.main import BCIGui, app, AlertMessageType, AlertMessageResponse, ScrollableFrame, LineItems

from bcipy.config import BCIPY_ROOT, DEFAULT_EXPERIMENT_PATH, EXPERIMENT_FILENAME
from bcipy.helpers.load import load_experiments, load_fields
from bcipy.helpers.save import save_experiment_data


class ExperimentRegistry(BCIGui):
    """Experiment Registry.

    User interface for creating new experiments for use in BCInterface.py.
    """

    padding = 100
    btn_height = 50
    default_text = '...'
    alert_title = 'Experiment Registry Alert'
    alert_timeout = 10
    experiment_fields = []

    def __init__(self, *args, **kwargs):
        super(ExperimentRegistry, self).__init__(*args, **kwargs)

        # Structure of an experiment:
        #   { name: { fields : {name: '', required: bool, anonymize: bool}, summary: '' } }
        self.update_experiment_data()

        # These are set in the build_inputs and represent text inputs from the user
        self.name_input = None
        self.summary_input = None
        self.field_input = None
        self.panel = None
        self.line_items = None

        # fields is for display of registered fields
        self.fields = []
        self.registered_fields = load_fields()
        self.name = None
        self.summary = None

        # for registered fields
        self.build_scroll_area()

        self.show_gui()
        self.update_field_list()

    def build_scroll_area(self) -> None:
        """Build Scroll Area.

        Appends a scrollable area at the bottom of the UI for management of registered fields via LineItems.
        """
        line_widget = LineItems([], self.width)
        self.panel = ScrollableFrame(200, self.width, background_color='white', widget=line_widget)
        self.add_widget(self.panel)

    def refresh_field_panel(self) -> None:
        """Refresh Field Panel.

        Reconstruct the line items from the registered fields and refresh the scrollable panel of registered fields.
        """
        self.build_line_items_from_fields()
        self.panel.refresh(self.line_items)

    def toggle_required_field(self) -> None:
        """Toggle Required Field.

        *Button Action*

        Using the field_name retrieved from the button (get_id), find the field in self.experiment_fields and toggle
            the required field ('true' or 'false').
        """
        field_name = self.window.sender().get_id()
        for field in self.experiment_fields:
            if field_name in field:
                required = field[field_name]['required']
                if required == 'false':
                    field[field_name]['required'] = 'true'
                else:
                    field[field_name]['required'] = 'false'
        self.refresh_field_panel()

    def toggle_anonymize_field(self) -> None:
        """Toggle Anonymize Field.

        *Button Action*

        Using the field_name retrieved from the button (get_id), find the field in self.experiment_fields and toggle
            the anonymize field ('true' or 'false').
        """
        field_name = self.window.sender().get_id()
        for field in self.experiment_fields:
            if field_name in field:
                anonymize = field[field_name]['anonymize']
                if anonymize == 'false':
                    field[field_name]['anonymize'] = 'true'
                else:
                    field[field_name]['anonymize'] = 'false'
        self.refresh_field_panel()

    def remove_field(self) -> None:
        """Remove Field.

        *Button Action*

        Using the field_name retrieved from the button (get_id), find the field in self.experiment_fields and remove it
            from the list.
        """
        field_name = self.window.sender().get_id()

        idx = 0
        remove = None
        for field in self.experiment_fields:
            if field_name in field:
                remove = idx
                break
            idx += 1

        self.experiment_fields.pop(remove)
        self.refresh_field_panel()

    def build_line_items_from_fields(self) -> None:
        """Build Line Items From Fields.

        Loop over the registered experiment fields and create LineItems, which can by used to toggle the required
            field, anonymization, or remove as a registered experiment field.
        """
        items = []
        for field in self.experiment_fields:
            # experiment fields is a list of dicts, here we loop over the dict to get
            # the field_name, anonymization, and requirement
            for field_name, required in field.items():

                # Set the button text and colors, based on the requirement and anonymization
                if required['required'] == 'false':
                    required_button_label = 'Optional'
                    required_button_color = 'black'
                    required_button_text_color = 'white'
                else:
                    required_button_label = 'Required'
                    required_button_color = 'green'
                    required_button_text_color = 'white'

                if required['anonymize'] == 'false':
                    anon_button_label = 'Onymous'
                    anon_button_color = 'black'
                    anon_button_text_color = 'white'
                else:
                    anon_button_label = 'Anonymous'
                    anon_button_color = 'green'
                    anon_button_text_color = 'white'

                # Construct the item to turn into a LineItem, we set the id as field_name to use later via the action
                item = {
                    field_name: {
                        required_button_label: {
                            'action': self.toggle_required_field,
                            'color': required_button_color,
                            'textColor': required_button_text_color,
                            'id': field_name
                        },
                        anon_button_label: {
                            'action': self.toggle_anonymize_field,
                            'color': anon_button_color,
                            'textColor': anon_button_text_color,
                            'id': field_name
                        },
                        'Remove': {
                            'action': self.remove_field,
                            'color': 'red',
                            'textColor': 'white',
                            'id': field_name
                        }
                    }
                }
                items.append(item)

        # finally, set the new line items for rendering
        self.line_items = LineItems(items, self.width)

    def build_text(self) -> None:
        """Build Text.

        Build all static text needed for the UI.
        Positions are relative to the height / width of the UI defined in start_app.
        """
        text_x = 25
        text_y = 70
        font_size = 18
        self.add_static_textbox(
            text='Create BciPy Experiment',
            position=[self.width / 2 - self.padding - 50, 0],
            size=[300, 100],
            background_color='black',
            text_color='white',
            font_size=22
        )
        self.add_static_textbox(
            text='Name',
            position=[text_x, text_y],
            size=[200, 50],
            background_color='black',
            text_color='white',
            font_size=font_size)
        text_y += self.padding
        self.add_static_textbox(
            text='Summary',
            position=[text_x, text_y],
            size=[300, 50],
            background_color='black',
            text_color='white',
            font_size=font_size)
        text_y += self.padding
        self.add_static_textbox(
            text='Fields',
            position=[text_x, text_y],
            size=[300, 50],
            background_color='black',
            text_color='white',
            font_size=font_size)
        text_y += self.padding + 45
        self.add_static_textbox(
            text='Registered fields *click to toggle required field*',
            position=[text_x, text_y],
            size=[300, 50],
            background_color='black',
            text_color='white',
            font_size=14)

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
            background_color='white',
            text_color='black')

        input_y += self.padding
        self.summary_input = self.add_combobox(
            position=[input_x, input_y],
            size=input_size,
            items=[self.default_text],
            editable=True,
            background_color='white',
            text_color='black')

        input_y += self.padding
        self.field_input = self.add_combobox(
            position=[input_x, input_y],
            size=input_size,
            items=self.fields,
            editable=False,
            background_color='white',
            text_color='black')

    def build_buttons(self):
        """Build Buttons.

        Build all buttons necessary for the UI. Define their action on click using the named argument action.
        """
        btn_create_x = self.width - self.padding - 10
        btn_create_y = self.height - self.padding - 200
        size = 150
        self.add_button(
            message='Create Experiment', position=[btn_create_x - (size / 2), btn_create_y],
            size=[size, self.btn_height],
            background_color='green',
            action=self.create_experiment,
            text_color='white')

        btn_field_x = (self.width / 2) + 150
        btn_field_y = 310
        # create field
        self.add_button(
            message='+',
            position=[btn_field_x, btn_field_y],
            size=[40, self.btn_height - 10],
            background_color='green',
            action=self.create_field,
            text_color='white'
        )

        # add field
        self.add_button(
            message='Register',
            position=[btn_field_x - 75, btn_field_y],
            size=[60, self.btn_height - 10],
            background_color='grey',
            action=self.add_field,
            text_color='white'
        )

    def create_experiment(self) -> None:
        """Create Experiment.

        After inputing all required fields, verified by check_input, add it to the experiment list and save it.
        """
        if self.check_input():
            self.add_experiment()
            self.save_experiments()
            self.throw_alert_message(
                title=self.alert_title,
                message='Experiment saved successfully! Please exit window or create another experiment!',
                message_type=AlertMessageType.INFO,
                message_response=AlertMessageResponse.OTE,
                message_timeout=self.alert_timeout
            )
            self.update_experiment_data()

    def update_experiment_data(self):
        """Update Experiment Data.

        Fetches the experiments and extracts the registered names.
        """
        self.experiments = load_experiments()
        self.experiment_names = self.experiments.keys()

    def add_experiment(self) -> None:
        """Add Experiment:

        Add a new experiment to the dict of experiments. It follows the format:
             { name: { fields : {name: '', required: bool, anonymize: bool}, summary: '' } }
        """
        self.experiments[self.name] = {
            'fields': self.experiment_fields,
            'summary': self.summary
        }

    def save_experiments(self) -> None:
        """Save Experiment.

        Save the experiments registered to the correct path as pulled from system_utils.
        """
        # add fields to the experiment
        save_experiment_data(self.experiments, self.registered_fields, DEFAULT_EXPERIMENT_PATH, EXPERIMENT_FILENAME)

    def create_field(self) -> None:
        """Create Field.

        Launch to FieldRegistry to create a new field for experiments.
        """
        subprocess.call(
            f'python {BCIPY_ROOT}/gui/experiments/FieldRegistry.py',
            shell=True)

        self.update_field_list()

    def add_field(self) -> None:
        """Add Field.

        Functionality to add fields to the newly created experiment. It will ensure no duplicates are addded.
        """
        # get the current field value and compute a list of field names already added
        field = self.field_input.currentText()
        registered_fields = [name for field in self.experiment_fields for name in field.keys()]

        # if the field selected is already registered throw an alert to the user
        if field in registered_fields:
            return self.throw_alert_message(
                title=self.alert_title,
                message=f'{field} already registered with this experiment!',
                message_type=AlertMessageType.INFO,
                message_response=AlertMessageResponse.OTE,
                message_timeout=self.alert_timeout,
            )

        # else add the field!
        self.experiment_fields.append(
            {
                field:
                    {
                        'required': 'false',
                        'anonymize': 'true'
                    }
            }
        )

        self.refresh_field_panel()

    def update_field_list(self) -> None:
        """Updates the field_input combo box with a list of fields. """

        self.field_input.clear()
        self.field_input.addItem(ExperimentRegistry.default_text)
        self.registered_fields = load_fields()
        self.fields = [item for item in self.registered_fields]
        self.field_input.addItems(self.fields)

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
        self.summary = self.summary_input.currentText()
        try:
            if self.name == ExperimentRegistry.default_text:
                self.throw_alert_message(
                    title=self.alert_title,
                    message='Please add an Experiment Name!',
                    message_type=AlertMessageType.WARN,
                    message_response=AlertMessageResponse.OTE,
                    message_timeout=self.alert_timeout)
                return False
            if self.name in self.experiment_names:
                self.throw_alert_message(
                    title=self.alert_title,
                    message=(
                        'Experiment name already registered. \n'
                        'Please use a unique Experiment name! \n'
                        f'Registed names: {self.experiment_names}'
                    ),
                    message_type=AlertMessageType.WARN,
                    message_response=AlertMessageResponse.OTE,
                    message_timeout=self.alert_timeout)
                return False
            if self.summary == ExperimentRegistry.default_text:
                self.throw_alert_message(
                    title=self.alert_title,
                    message='Please add an Experiment Summary!',
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
    """Start Experiment Registry."""
    bcipy_gui = app(sys.argv)
    ex = ExperimentRegistry(
        title='Experiment Registry',
        height=700,
        width=600,
        background_color='black')

    sys.exit(bcipy_gui.exec_())


if __name__ == '__main__':
    start_app()
