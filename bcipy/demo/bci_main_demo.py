# This is a demo of the main bci system. It will run the task defined here
#  using the parameters file passed to it.


def main():
    from bcipy.main import bcipy_main
    from bcipy.task import TaskType
    from bcipy.helpers.parameters import DEFAULT_PARAMETERS_PATH

    # Load a parameters file
    parameters = DEFAULT_PARAMETERS_PATH

    # Task. Ex. `RSVP Calibration`
    task = TaskType.by_value('RSVP Calibration')

    # Experiment. Use the default registered experiment!
    experiment = 'default'

    # Define a user
    user = 'bci_main_demo_user'

    bcipy_main(parameters, user, task, experiment)


if __name__ == "__main__":
    main()
