import os
import sys
import glob
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QListWidget, QPushButton, QWidget, QMessageBox, QLabel, 
    QLineEdit, QCheckBox
)
from PyQt6.QtCore import Qt
from bcipy.core.validate import BciPyDataValidator
from bcipy.config import (
    DEFAULT_DEVICES_PATH,
    DEFAULT_DEVICE_SPEC_FILENAME,
    RAW_DATA_FILENAME,
    TRIGGER_FILENAME,
    DEFAULT_PARAMETERS_PATH)
from bcipy.io.load import load_experimental_data, load_raw_data, load_json_parameters
from bcipy.core.triggers import trigger_decoder, TriggerType
from bcipy.helpers.acquisition import analysis_channels
from bcipy.core.stimuli import InquiryReshaper, update_inquiry_timing
from bcipy.core.parameters import Parameters
from bcipy.signal.process import (ERPTransformParams,
                                  filter_inquiries, get_default_transform)

import bcipy.acquisition.devices as devices
devices_by_name = devices.load(
        Path(DEFAULT_DEVICES_PATH, DEFAULT_DEVICE_SPEC_FILENAME), replace=True)
logging.basicConfig(
            level=logging.DEBUG, 
            format='%(levelname)s: %(message)s'
        )

class DirectoryFinder(QMainWindow):
    def __init__(
        self, 
        root_directory, 
        default_search_pattern="*Calibration*", 
        validate_bcipy=False
    ):
        """
        Initialize the Directory Finder
        
        Args:
            root_directory (str): Starting directory for search
            default_search_pattern (str, optional): Glob pattern to search for directories
            validate_bcipy (bool, optional): Whether to validate BciPy data directories
        """
        super().__init__()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(levelname)s: %(message)s'
        )
        
        self.root_directory = os.path.abspath(root_directory)
        self.default_search_pattern = default_search_pattern
        self.validate_bcipy = validate_bcipy
        self.found_directories = []
        
        # Initialize UI
        self.init_ui()
        
        # Find directories
        self.find_directories()
        
    def find_directories(self):
        """
        Search for directories using glob pattern with optional BciPy validation
        """
        # Construct full glob pattern
        full_pattern = os.path.join(
            self.root_directory, 
            '**' if self.search_recursive_checkbox.isChecked() else '', 
            self.pattern_input.text()
        )
        
        # Perform directory search
        try:
            # Using glob.glob with recursive option
            potential_dirs = [
                path for path in glob.glob(full_pattern, recursive=self.search_recursive_checkbox.isChecked()) 
                if os.path.isdir(path)
            ]
            
            # Filter directories based on BciPy validation if enabled
            if self.validate_bcipy_checkbox.isChecked():
                self.found_directories = [
                    path for path in potential_dirs
                    if BciPyDataValidator.validate_bcipy_directory(path)
                ]
            else:
                self.found_directories = potential_dirs
        
        except Exception as e:
            QMessageBox.warning(
                self, 
                "Search Error", 
                f"An error occurred during search: {str(e)}"
            )
            self.found_directories = []
        
        # Update found files indicator
        self.found_files_label.setText(f"Found: {len(self.found_directories)} directories")
        
        # Populate list widget
        self.directory_list.clear()
        self.directory_list.addItems(self.found_directories)
        
        # Select all items by default
        self.select_all()
        
        # Show message if no directories found
        if not self.found_directories:
            QMessageBox.information(
                self, 
                "No Directories", 
                f"No directories matching '{self.pattern_input.text()}' were found."
            )
        
    def init_ui(self):
        """
        Create the user interface
        """
        # Set up main window
        self.setWindowTitle("Directory Finder (BciPy Aware)")
        self.resize(800, 600)
        
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Search pattern input
        search_layout = QHBoxLayout()
        search_label = QLabel("Glob Pattern:")
        self.pattern_input = QLineEdit(self.default_search_pattern)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.pattern_input)
        main_layout.addLayout(search_layout)
        
        # Help text for glob patterns
        help_label = QLabel(
            "Glob Pattern Help: \n"
            "* matches any number of characters\n"
            "? matches any single character\n"
            "[seq] matches any character in seq\n"
            "[!seq] matches any character not in seq"
        )
        help_label.setWordWrap(True)
        main_layout.addWidget(help_label)
        
        # Options layout
        options_layout = QHBoxLayout()
        
        # Recursive search checkbox
        self.search_recursive_checkbox = QCheckBox("Recursive Search")
        self.search_recursive_checkbox.setChecked(True)
        options_layout.addWidget(self.search_recursive_checkbox)
        
        # BciPy validation checkbox
        self.validate_bcipy_checkbox = QCheckBox("Validate BciPy Directories")
        self.validate_bcipy_checkbox.setChecked(self.validate_bcipy)
        options_layout.addWidget(self.validate_bcipy_checkbox)
        
        # Found files label
        self.found_files_label = QLabel("Found: 0 directories")
        options_layout.addWidget(self.found_files_label)
        
        # Search button
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.find_directories)
        options_layout.addWidget(search_button)
        
        main_layout.addLayout(options_layout)
        
        # Title label
        title_label = QLabel("Select Directories")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Directory list
        self.directory_list = QListWidget()
        self.directory_list.setSelectionMode(
            QListWidget.SelectionMode.MultiSelection
        )
        main_layout.addWidget(self.directory_list)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # Select All button
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        button_layout.addWidget(select_all_btn)
        
        # Deselect All button
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self.deselect_all)
        button_layout.addWidget(deselect_all_btn)
        
        # Confirm button
        confirm_btn = QPushButton("Confirm Selection")
        confirm_btn.clicked.connect(self.confirm_selection)
        button_layout.addWidget(confirm_btn)
        
        # Add button layout to main layout
        main_layout.addLayout(button_layout)
        
        # Set layout on central widget
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Selected paths will be stored here
        self.selected_paths = []
        
    def select_all(self):
        """Select all directories in the list"""
        for i in range(self.directory_list.count()):
            self.directory_list.item(i).setSelected(True)
        
    def deselect_all(self):
        """Deselect all directories in the list"""
        for i in range(self.directory_list.count()):
            self.directory_list.item(i).setSelected(False)
        
    def confirm_selection(self):
        """
        Process the selected directories and close the window
        """
        # Get selected items
        selected_items = self.directory_list.selectedItems()
        
        # Collect selected paths
        self.selected_paths = [item.text() for item in selected_items]
        
        # Close the window
        self.close()

def find_directories(
    start_directory, 
    search_pattern="*Calibration*", 
    recursive=True,
    validate_bcipy=False
):
    """
    Main function to find and select directories using glob patterns
    
    Args:
        start_directory (str): Directory to start searching from
        search_pattern (str, optional): Glob pattern to search for
        recursive (bool, optional): Whether to search recursively
        validate_bcipy (bool, optional): Whether to validate BciPy data directories
    
    Returns:
        list: List of selected directory paths
    """
    # Create Qt Application
    app = QApplication(sys.argv)
    
    # Create and show the finder
    finder = DirectoryFinder(
        start_directory, 
        search_pattern,
        validate_bcipy
    )
    finder.search_recursive_checkbox.setChecked(recursive)
    finder.show()
    
    # Run the application
    app.exec()
    
    return finder.selected_paths

def load_group_trials(directories: List[str], parameters: Parameters) -> Tuple[List[np.ndarray], List[int]]:
    """Given a list of directories, load the group trials for training a model.
    
    It should return two lists: one containing the trials and the other containing the labels.

    Args:
        directories (list): List of directories containing valid experimental data.
        parameters (Parameters): Parameters object containing the necessary parameters for loading trials.

    Returns:
        tuple: A tuple containing the list of trials and the list of labels.
    """
    all_trials = []
    all_labels = []

     # Extract relevant session information from parameters file
    trial_window = parameters.get("trial_window")
    window_length = trial_window[1] - trial_window[0]

    prestim_length = parameters.get("prestim_length")
    trials_per_inquiry = parameters.get("stim_length")
    # The task buffer length defines the min time between two inquiries
    # We use half of that time here to buffer during transforms
    buffer = int(parameters.get("task_buffer_length") / 2)

    # Get signal filtering information
    transform_params = parameters.instantiate(ERPTransformParams)
    downsample_rate = transform_params.down_sampling_rate
    reshaper = InquiryReshaper()
    
    
    for directory in directories:
        # load raw data, triggers, and parameters
        erp_data = load_raw_data(f"{directory}/{RAW_DATA_FILENAME}.csv")
        device_spec = devices_by_name.get(erp_data.daq_type)
        static_offset = device_spec.static_offset
        channels = erp_data.channels
        sample_rate = erp_data.sample_rate

        # setup filtering
        default_transform = get_default_transform(
            sample_rate_hz=sample_rate,
            notch_freq_hz=transform_params.notch_filter_frequency,
            bandpass_low=transform_params.filter_low,
            bandpass_high=transform_params.filter_high,
            bandpass_order=transform_params.filter_order,
            downsample_factor=transform_params.down_sampling_rate,
        )

        # Process triggers.txt files
        trigger_targetness, trigger_timing, _ = trigger_decoder(
            trigger_path=f"{directory}/{TRIGGER_FILENAME}",
            exclusion=[TriggerType.PREVIEW, TriggerType.EVENT],
            offset=static_offset,
            remove_pre_fixation=True,
            device_type='EEG'
        )

        # update the trigger timing list to account for the initial trial window
        corrected_trigger_timing = [timing + trial_window[0] for timing in trigger_timing]

        # Channel map can be checked from raw_data.csv file or the devices.json located in the acquisition module
        # The timestamp column [0] is already excluded.
        channel_map = analysis_channels(channels, device_spec)
        channels_used = [channels[i] for i, keep in enumerate(channel_map) if keep == 1]
        logging.info(f'Channels used in analysis: {channels_used}')

        data, fs = erp_data.by_channel()

        inquiries, inquiry_labels, inquiry_timing = reshaper(
            trial_targetness_label=trigger_targetness,
            timing_info=corrected_trigger_timing,
            eeg_data=data,
            sample_rate=sample_rate,
            trials_per_inquiry=trials_per_inquiry,
            channel_map=channel_map,
            poststimulus_length=window_length,
            prestimulus_length=prestim_length,
            transformation_buffer=buffer,
        )

        inquiries, fs = filter_inquiries(inquiries, default_transform, sample_rate)
        inquiry_timing = update_inquiry_timing(inquiry_timing, downsample_rate)
        trial_duration_samples = int(window_length * fs)
        data = reshaper.extract_trials(inquiries, trial_duration_samples, inquiry_timing)

        # define the training classes using integers, where 0=nontargets/1=targets
        labels = inquiry_labels.flatten().tolist()

        all_trials.extend(data)
        all_labels.extend(labels)
        
    return all_trials, all_labels

def train_model(trials: List[np.ndarray], labels: List[int]):
    """
    Train a model using the given trials and labels.
    
    Args:
        trials (list): List of trials to train the model on
        labels (list): List of labels corresponding to the trials
    """
    # Insert model training code here
    logging.info("Model training not implemented yet.")
    pass

# Example usage
if __name__ == "__main__":
    # Replace with the directory you want to search
    start_dir = load_experimental_data()
    
    # Find and select directories
    selected_dirs = find_directories(
        start_dir,
        search_pattern="*Calibration*",
        recursive=True,
        validate_bcipy=True
    )
    
    # Print selected directories
    print("Selected Directories:")
    for path in selected_dirs:
        print(path)

    # parameters_path = selected_dirs[0] + "/" + DEFAULT_PARAMETERS_FILENAME
    parameters_path = DEFAULT_PARAMETERS_PATH
    logging.info(f"Loading parameters from {parameters_path}")
    # Load group trials
    parameters = load_json_parameters(parameters_path, True)
    
    trials, labels = load_group_trials(selected_dirs, parameters)
    logging.info(
        f"Loaded {len(trials)} trials and {len(labels)} labels. Accrued from {len(selected_dirs)} directories.")

    train_model(trials, labels)