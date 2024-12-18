import os
import glob

import logging
from bcipy.config import RAW_DATA_FILENAME, TRIGGER_FILENAME, SESSION_DATA_FILENAME, DEFAULT_PARAMETERS_FILENAME

logging.basicConfig(
            level=logging.WARNING, 
            format='%(levelname)s: %(message)s'
        )

MIN_TRIGGER_LINES = 3


class BciPyDataValidator:
    """
    Utility class to validate BciPy data directory structure
    """

    @staticmethod
    def validate_bcipy_directory(directory: str, online: bool=False, contain_model: bool=False) -> bool:
        """
        Validate if a directory contains valid BciPy data files
        
        Args:
            directory (str): Path to directory to validate.
                The directory should contain raw_data.csv, triggers.txt and optionally session.json and model file.
            online (bool): Whether to check for online data files. Defaults to False.
                This will check for session.json file in the directory.
            contain_model (bool): Whether to check for model file. Defaults to False.
                This will check for a single pkl file in the directory.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # Check if directory exists
            if not os.path.isdir(directory):
                logging.warning(f"Invalid directory: {directory}")
                return False
            
            # Check for raw data file
            raw_data_path = os.path.join(directory, f"{RAW_DATA_FILENAME}.csv")
            if not os.path.exists(raw_data_path):
                logging.warning(f"Missing {RAW_DATA_FILENAME} in: {directory}")
                return False
            
            # Check for triggers file
            triggers_path = os.path.join(directory, TRIGGER_FILENAME)
            if not os.path.exists(triggers_path):
                logging.warning(f"Missing {TRIGGER_FILENAME} in: {directory}")
                return False
            else:
                with open(triggers_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) < MIN_TRIGGER_LINES:
                        logging.warning(f"Empty {TRIGGER_FILENAME} in: {directory}")
                        return False
            
            # check for parameters file
            parameters_path = os.path.join(directory, DEFAULT_PARAMETERS_FILENAME)
            if not os.path.exists(parameters_path):
                logging.warning(f"Missing {DEFAULT_PARAMETERS_FILENAME} in: {directory}")
                return False
            
            if online:
                # Check for session.json
                session_path = os.path.join(directory, SESSION_DATA_FILENAME)
                if not os.path.exists(session_path):
                    logging.warning(f"Missing {SESSION_DATA_FILENAME} in: {directory}")
                    return False
                
            if contain_model:
                # search for a pkl file
                model_files = glob.glob(os.path.join(directory, "*.pkl"))
                if not model_files:
                    logging.warning(f"Missing .pkl model file in: {directory}")
                    return False
                
                if len(model_files) > 1:
                    logging.warning(f"Found multiple model files in: {directory}. Expected only one.")
                    return False
                
        except Exception as e:
            logging.error(f"Error validating directory: {directory}. {str(e)}")
            return False

        return True
