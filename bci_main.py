from helpers.save import init_save_data_structure
from display.display_main import init_display_window
from helpers.acquisition_related import init_eeg_acquisition

from bci_tasks.start_task import start_task
from helpers.load import load_classifier
from helpers.lang_model_related import init_language_model


def bci_main(parameters, user, exp_type, mode):
    """BCI Main.

    The BCI main function will initialize a save folder, construct needed information
    and execute the task. This is the main connection between any UI and
    running the app.

    It may also be invoked via tha command line.
        Ex. `python bci_main.py` this will default parameters, mode, user, and type.

        You can pass it those attributes with flags, if desired.
            Ex. `python bci_main.py --user "bci_user" --mode "SHUFFLE"`

    Input:
        parameters (dict): parameter dictionary
        user (str): name of the user
        exp_type (int): type of experiment. Ex. 1 = calibration
        mode (str): BCI mode. Ex. RSVP, SHUFFLE, MATRIX

    """

    # Define the parameter and data save location
    parameter_location = parameters['parameter_location']
    data_save_location = parameters['data_save_loc']

    # Initialize Save Folder
    save_folder = init_save_data_structure(
        data_save_location, user, parameter_location)

    # Register Task Type
    task_type = {
        'mode': mode,
        'exp_type': exp_type
    }

    # Try executing the task
    try:
        execute_task(
            task_type, parameters, save_folder)

    # Something went wrong, raise exception to caller
    except Exception as e:
        raise e


def execute_task(task_type, parameters, save_folder):
    """
    Excecute Task.

    Executes the desired task by setting up the display window and
        data acquistion, then passing on to the start_task funtion
        which will initialize experiment.

    Input:
        parameters (dict): parameter dictionary
        task_type (dict): type and mode of experiment
        save_folder (str): path to save folder
    """

    fake = parameters['fake_data']

    # Init EEG Model, if needed. Calibration Tasks Don't require probalistic
    #   modules to be loaded.
    if task_type['exp_type'] > 1:

        # Try loading in our classifier and starting a langmodel(if enabled)
        try:

            # EEG Model, Load in pre-trained classifier
            if fake:
                classifier = None
            else:
                classifier = load_classifier()

            # if Language Model enabled and data not fake, init lm
            if parameters['languagemodelenabled'] == 'true' \
                    and not fake:
                try:
                    lmodel = init_language_model(parameters)
                except:
                    print("Cannot init language model. Setting to None.")
                    lmodel = None
            else:
                lmodel = None

        except Exception as e:
            print("Cannot load EEG classifier. Exiting")
            raise e

    else:
        classifier = None
        lmodel = None

    # Initialize DAQ
    daq, server = init_eeg_acquisition(
        parameters, save_folder, server=fake)

    # Initialize Display Window
    display = init_display_window(parameters)

    # Start Task
    try:
        start_task(
            display, daq, task_type, parameters, save_folder,
            lmodel=lmodel,
            classifier=classifier, fake=fake)

    # If exception, close all display and acquisition objects
    except Exception as e:
        # close display
        display.close()
        raise e

    # Close the display window
    display.close()

    # Stop Acquisition
    daq.stop_acquisition()
    daq.cleanup()

    if server:
        server.stop()

    return


if __name__ == "__main__":
    import argparse
    from helpers.load import load_json_parameters
    import multiprocessing

    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameters', default='parameters/parameters.json',
                        help='Parameter location. Must be in parameters directory. Pass as parameters/parameters.json')
    parser.add_argument('-u', '--user', default='test_user')
    parser.add_argument('-t', '--type', default=1,
                        help='Task Type for a given mode. Ex. RSVP, 1 is calibration')
    parser.add_argument('-m', '--mode', default='RSVP',
                        help='BCI mode. Ex. RSVP, MATRIX, SHUFFLE')

    args = parser.parse_args()

    # Load a parameters file
    parameters = load_json_parameters(args.parameters, value_cast=True)

    # Start BCI Main
    bci_main(parameters, str(args.user), int(args.type), str(args.mode))
