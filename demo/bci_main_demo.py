# This is a demo of the main bci system. It will run the task defined here
#  using the parameters file passed to it.


def main():
    import bci_main
    from bcipy.helpers.load import load_json_parameters
    from bcipy.tasks.task_registry import TaskType
    from bcipy.helpers.parameters import DEFAULT_PARAMETERS_PATH

    # Load a parameters file
    parameters = DEFAULT_PARAMETERS_PATH

    # Task. Ex. `RSVP Calibration`
    task = TaskType.by_value('RSVP Calibration')

    # Experiment. Use the default registered experiment!
    experiment = 'default'

    # Define a user
    user = 'bci_main_demo_user'

    # Try and initialize with bci main
    try:
        bci_main.bci_main(parameters, user, task, experiment)
    except Exception as e:
        print("BCI MAIN Fail. Exiting. Error: \n")
        print(e)


if __name__ == "__main__":
    main()
