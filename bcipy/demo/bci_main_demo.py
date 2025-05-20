from bcipy.config import DEFAULT_PARAMETERS_PATH
from bcipy.main import bci_main

parameter_location = DEFAULT_PARAMETERS_PATH  # Path to a valid BciPy parameters file
user = 'test_demo_user'  # User ID
experiment_id = 'default'  # This will run two tasks: RSVP Calibration and Matrix Calibration
alert = False  # Set to True to alert user when tasks are complete
visualize = False  # Set to True to visualize data at the end of a task
fake_data = True  # Set to True to use fake acquisition data during the session
# A single task or experiment ID must be provided to run. If a task is provided, the experiment ID will be ignored.
task = None


def bcipy_main():
    """BCI Main Demo.

    This function demonstrates how to use the BciPy main function outside of the client interface to execute tasks
      or experiments.
    """
    bci_main(
        parameter_location=parameter_location,
        user=user,
        experiment_id=experiment_id,
        alert=alert,
        visualize=visualize,
        fake=fake_data,
        task=task
    )


if __name__ == '__main__':
    bcipy_main()
