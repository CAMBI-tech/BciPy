# -*- coding: utf-8 -*-
import errno
import os
from time import localtime, strftime
from shutil import copy2
import json


def init_save_data_structure(data_save_path,
                             user_information,
                             parameters_used):
    """
    Initialize Save Data Strucutre.

        data_save_path[str]: string of path to save our data in
        user_information[str]: string of user name / realted information
        parameters_used[str]: a path to parameters file for the experiment

    """

    # make an experiment folder : note datetime is in utc
    save_folder_name = data_save_path + user_information
    save_folder_run_name = save_folder_name + '/' + \
        user_information + '_' + strftime(
            '%a_%d_%b_%Y_%Hhr%Mmin%Ssec_%z', localtime())
    helper_folder_name = save_folder_run_name + '/helpers/'

    # try making the given path
    try:
        # make a directory to save data to
        os.makedirs(save_folder_name)
        os.makedirs(save_folder_run_name)
        os.makedirs(helper_folder_name)

    except OSError as error:
        # If the error is anything other than file existing, raise an error
        if error.errno != errno.EEXIST:
            raise error

        # since this is only called on init, we can make another folder run
        os.makedirs(save_folder_run_name)
        os.makedirs(helper_folder_name)

    try:
        # Go to folder helpers and list files within it
        src_files = os.listdir('bcipy/helpers/')

        # Loop through files in helpers and copy important ones over
        for file_name in src_files:

            # get the full name
            full_file_name = os.path.join('bcipy/helpers/', file_name)

            # Check that constructed file is a real file and ends in .py
            if (os.path.isfile(full_file_name)) and '.py' in file_name:
                # Copy over!
                copy2(full_file_name, helper_folder_name)

        # Check that parameters file given is a real file
        import pdb;
        pdb.set_trace()
        if (os.path.isfile(parameters_used)):
            # Copy over parameters file
            copy2(parameters_used, save_folder_run_name)
        else:
            raise Exception('Parameter File Not Found!')

    # catch IO exceptions
    except IOError as error:
        raise error

    except Exception as error:
        raise error

    return save_folder_run_name


def _save_session_related_data(file, session_dictionary):
    """
    Save Session Related Data.

    Parameters
    ----------
        file[str]: string of path to save our data in
        session_dictionary[dict]: dictionary of session data. It will appear
            as follows:
                {{ "epochs": {
                        "1": {
                          "0": {
                            "copy_phrase": "COPY_PHRASE",
                            "current_text": "COPY_",
                            "eeg_len": 22,
                            "next_display_state": "COPY_",
                            "stimuli": [["+", "_", "G", "L", "B"]],
                            "target_info": [
                              "nontarget", ... ,
                            ],
                            "timing_sti": [[1, 0.2, 0.2, 0.2, 0.2]],
                            "triggers": [[ "+", 0.0], ["_", 0.9922] ..],
                            },
                        ... ,
                        "7": {
                            ... ,
                  },
                  "paradigm": "RSVP",
                  "session": "data/demo_user/demo_user",
                  "session_type": "Copy Phrase",
                  "total_time_spent": 83.24798703193665
                }}
    Returns
    -------
        file, session data file (json file)

    """
    # Try opening as json, if not able to use open() to create first
    try:
        file = json.load(file, 'wt')
    except:
        file = open(file, 'wt')

    # Use the file to dump data to
    try:
        json.dump(session_dictionary, file, indent=2)
    except Exception as e:
        raise e

    return file
