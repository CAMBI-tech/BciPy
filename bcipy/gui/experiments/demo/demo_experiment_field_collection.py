from bcipy.core.session import collect_experiment_field_data
from bcipy.config import DEFAULT_EXPERIMENT_ID


experiment_name = DEFAULT_EXPERIMENT_ID
# this will save the data at the location you've run the demo from!
save_folder = '.'
collect_experiment_field_data(experiment_name, save_folder)
