"""
Transfer Learning Group Model Training and Evaluation

This script provides functionality to:
- Find and select BciPy calibration directories
- Load and preprocess group trials from multiple sessions
- Train signal models (PcaRdaKdeModel or EEGNet)
- Evaluate models using leave-one-subject-out cross-validation
- Compute comprehensive performance metrics

TODO
- EEGNet into BciPy SignalModel wrapper
- A way to save the model during adaptive copy phrase
- Online evaluation of signal model 
- Get data to Basak 

Decisions:
- The model will handle its updating and state tracking

Timeline:
- 2 weeks meet (29th) to discuss the model and the data
"""

import os
import sys
import glob
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
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
from bcipy.io.save import save_model
from bcipy.helpers.acquisition import analysis_channels
from bcipy.core.stimuli import InquiryReshaper, update_inquiry_timing
from bcipy.core.parameters import Parameters
from bcipy.signal.process import (ERPTransformParams,
                                  filter_inquiries, get_default_transform)
from bcipy.signal.model import PcaRdaKdeModel, SignalModel
from eeg_net import EEGNet
from bcipy.signal.model.base_model import SignalModelMetadata
from bcipy.helpers.utils import report_execution_time

import bcipy.acquisition.devices as devices
from bcipy.acquisition.devices import DeviceSpec
from bcipy.signal.process import Composition
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils as np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    balanced_accuracy_score
)

devices_by_name = devices.load(
        Path(DEFAULT_DEVICES_PATH, DEFAULT_DEVICE_SPEC_FILENAME), replace=True)

GROUP_MODEL_NAME = "PCA_RSVP_ME_group_model_2"

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

def load_group_trials(
        directories: List[str],
        parameters: Parameters,
        external_model: bool = True,
        return_per_subject: bool = False) -> Tuple:
    """Given a list of directories, load the group trials for training a model.
    
    It should return two lists: one containing the trials and the other containing the labels.

    Args:
        directories (list): List of directories containing valid experimental data.
        parameters (Parameters): Parameters object containing the necessary parameters for loading trials.
        external_model (bool, optional): Whether to use an external model. Defaults to False.
            The default will return the data in (Channels X Trials X Samples).
            If True, the trials will be transposed before returning into (Trials X Channels x Samples).
        return_per_subject (bool, optional): If True, returns data organized by subject. Defaults to False.

    Returns:
        tuple: A tuple containing the list of trials, labels, device_spec, and transform.
               If return_per_subject is True, also returns a list of subject indices.
    """
    all_trials = []
    all_labels = []
    subject_indices = []  # Track which trials belong to which subject

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
    
    
    for subject_idx, directory in enumerate(directories):
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
        print(f'Subject {subject_idx + 1}/{len(directories)} - Channels used in analysis: {channels_used}')

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

        if external_model:
            data = np.transpose(data, (1, 0, 2))
        
        # Track subject indices for LOSO cross-validation
        n_trials = len(labels)
        subject_indices.extend([subject_idx] * n_trials)
        
        all_trials.append(data)
        all_labels.extend(labels)
    
    if not external_model:
        all_trials = np.concatenate(all_trials, axis=1)
    else:
        all_trials = np.concatenate(all_trials, axis=0)
    
    if return_per_subject:
        return all_trials, all_labels, device_spec, default_transform, subject_indices
    return all_trials, all_labels, device_spec, default_transform

def compute_evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for binary classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for ROC-AUC)
    
    Returns:
        Dictionary containing various performance metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['sensitivity'] = metrics['recall']  # Same as recall for positive class
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # ROC-AUC if probabilities are provided
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['roc_auc'] = 0.0
    
    return metrics


def print_evaluation_summary(metrics: Dict[str, float], fold_name: str = ""):
    """
    Print a formatted summary of evaluation metrics.
    
    Args:
        metrics: Dictionary of evaluation metrics
        fold_name: Optional name for the fold/iteration
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Results {fold_name}")
    print(f"{'='*60}")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall/Sensitivity: {metrics['recall']:.4f}")
    print(f"Specificity:        {metrics['specificity']:.4f}")
    print(f"F1 Score:           {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['true_positives']:4d}  FN: {metrics['false_negatives']:4d}")
    print(f"  FP: {metrics['false_positives']:4d}  TN: {metrics['true_negatives']:4d}")
    print(f"{'='*60}\n")


@report_execution_time
def train_model(
        trials: np.ndarray,
        labels: List[int],
        device_spec: dict = None,
        default_transform: dict = None,
        model_type: str = "PcaRdaKdeModel") -> SignalModel:
    """
    Train a model using the given trials and labels.

    TODO implement a scikit-learn model to train on the trials and labels
    
    Args:
        trials (list): List of trials to train the model on
        labels (list): List of labels corresponding to the trials
        device_spec (dict, optional): Device specification dictionary
        default_transform (dict, optional): Default transformation dictionary
    """
    if model_type == "PcaRdaKdeModel":
        # Data is in the form (Channels X Trials X Samples)
        # Insert model training code here
        model = PcaRdaKdeModel()
        print("Training model... This may take a while.")
        model.fit(trials, labels)
        model.metadata = SignalModelMetadata(
            device_spec=device_spec,
            transform=default_transform,
            evidence_type="ERP",
            auc=model.auc)
        save_model(model, f"{GROUP_MODEL_NAME}_{model.auc:0.4f}.pkl")
    
    if model_type == "EEGNet":
        # Split the data into training, testing, and validation sets (80/10/10)
        chans, samples, kernels = trials.shape[1], trials.shape[2], 1
        print(f"Channels: {chans}, Samples: {samples}, Kernels: {kernels}")
        X_train, X_test, Y_train, Y_test = train_test_split(trials, labels, test_size=0.2, random_state=42)
        X_validate, X_test, Y_validate, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)
        X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
        X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
        X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)
        
        # make labels np.array
        Y_test       = np.array(Y_test)
        Y_train      = np.array(Y_train)
        Y_validate   = np.array(Y_validate)
        Y_train      = Y_train.reshape(Y_train.shape[0], 1)
        Y_validate   = Y_validate.reshape(Y_validate.shape[0], 1)
        Y_test       = Y_test.reshape(Y_test.shape[0], 1)
  
        model = EEGNet(nb_classes=2, Chans=chans, Samples=samples)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # count number of parameters in the model
        numParams    = model.count_params()
        print(f"Number of parameters in model: {numParams}")
        class_weights = {0:1, 1:1}

        # set a valid path for your system to record model checkpoints
        checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.keras', verbose=1,
                                    save_best_only=True)
        fittedModel = model.fit(X_train, Y_train, epochs = len(labels), batch_size = 16,
                        verbose = 2, validation_data=(X_validate, Y_validate),
                        callbacks=[checkpointer], class_weight = class_weights)


        model.load_weights('/tmp/checkpoint.keras')
        probs       = model.predict(X_test)
        preds       = probs.argmax(axis = -1)  
        acc         = np.mean(preds == Y_test.argmax(axis=-1))
        print("Classification accuracy: %f " % (acc))
        breakpoint()
    return model


@report_execution_time
def leave_one_subject_out_evaluation(
        trials: np.ndarray,
        labels: List[int],
        subject_indices: List[int],
        device_spec: dict = None,
        default_transform: dict = None,
        model_type: str = "PcaRdaKdeModel",
        save_models: bool = False) -> Dict:
    """
    Perform leave-one-subject-out (LOSO) cross-validation.
    
    This evaluates model generalization across subjects by:
    - Training on N-1 subjects
    - Testing on the held-out subject
    - Repeating for all subjects
    
    Args:
        trials: EEG trial data
        labels: Trial labels
        subject_indices: List indicating which subject each trial belongs to
        device_spec: Device specification dictionary
        default_transform: Default transformation dictionary
        model_type: Type of model to train ("PcaRdaKdeModel" or "EEGNet")
        save_models: Whether to save trained models
    
    Returns:
        Dictionary containing LOSO results and aggregated metrics
    """
    # Convert to numpy arrays
    trials = np.array(trials)
    labels = np.array(labels)
    subject_indices = np.array(subject_indices)
    
    unique_subjects = np.unique(subject_indices)
    n_subjects = len(unique_subjects)
    
    print(f"\n{'='*60}")
    print(f"Leave-One-Subject-Out Cross-Validation")
    print(f"Total Subjects: {n_subjects}")
    print(f"Model Type: {model_type}")
    print(f"{'='*60}\n")
    
    fold_results = []
    all_true_labels = []
    all_predictions = []
    all_probabilities = []
    
    for fold, test_subject in enumerate(unique_subjects):
        print(f"\nFold {fold + 1}/{n_subjects}: Testing on Subject {test_subject}")
        
        # Split data
        test_mask = subject_indices == test_subject
        train_mask = ~test_mask
        
        X_train = trials[train_mask]
        y_train = labels[train_mask]
        X_test = trials[test_mask]
        y_test = labels[test_mask]
        
        print(f"  Training samples: {len(y_train)} ({np.sum(y_train)} targets)")
        print(f"  Testing samples:  {len(y_test)} ({np.sum(y_test)} targets)")
        
        # Train model
        if model_type == "PcaRdaKdeModel":
            # Transpose for PcaRdaKdeModel (expects Channels X Trials X Samples)
            X_train_formatted = np.transpose(X_train, (1, 0, 2))
            
            model = PcaRdaKdeModel()
            print("  Training PcaRdaKdeModel...")
            model.fit(X_train_formatted, y_train.tolist())
            model.metadata = SignalModelMetadata(
                device_spec=device_spec,
                transform=default_transform,
                evidence_type="ERP",
                auc=model.auc if hasattr(model, 'auc') else 0.0
            )
            
            # Predict on test set
            X_test_formatted = np.transpose(X_test, (1, 0, 2))
            try:
                # Get probability estimates
                y_prob = model.predict_proba(X_test_formatted)
                if y_prob.ndim == 2:
                    y_prob = y_prob[:, 1]  # Get probability of positive class
                y_pred = (y_prob > 0.5).astype(int)
            except AttributeError:
                # Fallback if predict_proba not available
                y_pred = model.predict(X_test_formatted)
                y_prob = None
            
            if save_models:
                save_model(model, f"{GROUP_MODEL_NAME}_LOSO_fold{fold+1}_subj{test_subject}.pkl")
        
        elif model_type == "EEGNet":
            # Format data for EEGNet
            chans, samples = X_train.shape[1], X_train.shape[2]
            kernels = 1
            
            X_train_formatted = X_train.reshape(X_train.shape[0], chans, samples, kernels)
            X_test_formatted = X_test.reshape(X_test.shape[0], chans, samples, kernels)
            
            # Further split training into train/validation
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train_formatted, y_train, test_size=0.1, random_state=42
            )
            
            model = EEGNet(nb_classes=2, Chans=chans, Samples=samples)
            model.compile(optimizer='adam', 
                         loss='sparse_categorical_crossentropy', 
                         metrics=['accuracy'])
            
            print(f"  Training EEGNet (params: {model.count_params()})...")
            
            # Train with early stopping
            checkpoint_path = f'/tmp/loso_fold{fold+1}_checkpoint.keras'
            checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=0,
                                          save_best_only=True)
            
            model.fit(X_train_split, y_train_split, 
                     epochs=min(100, len(y_train)),  # Adaptive epochs
                     batch_size=16,
                     verbose=0,
                     validation_data=(X_val, y_val),
                     callbacks=[checkpointer])
            
            # Load best model and predict
            model.load_weights(checkpoint_path)
            y_prob = model.predict(X_test_formatted, verbose=0)
            y_pred = y_prob.argmax(axis=-1)
            y_prob = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob.flatten()
            
            if save_models:
                model.save(f"{GROUP_MODEL_NAME}_LOSO_fold{fold+1}_subj{test_subject}.keras")
        
        # Compute metrics for this fold
        fold_metrics = compute_evaluation_metrics(y_test, y_pred, y_prob)
        fold_metrics['fold'] = fold + 1
        fold_metrics['test_subject'] = int(test_subject)
        fold_metrics['n_train'] = int(len(y_train))
        fold_metrics['n_test'] = int(len(y_test))
        fold_results.append(fold_metrics)
        
        # Accumulate for overall metrics
        all_true_labels.extend(y_test)
        all_predictions.extend(y_pred)
        if y_prob is not None:
            all_probabilities.extend(y_prob)
        
        # Print fold results
        print_evaluation_summary(fold_metrics, f"(Fold {fold + 1}, Subject {test_subject})")
    
    # Compute overall metrics across all folds
    all_true_labels = np.array(all_true_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities) if all_probabilities else None
    
    overall_metrics = compute_evaluation_metrics(
        all_true_labels, all_predictions, all_probabilities
    )
    
    # Compute mean and std across folds
    metric_names = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 
                    'specificity', 'f1_score', 'roc_auc']
    fold_stats = {}
    for metric in metric_names:
        if metric in fold_results[0]:
            values = [fr[metric] for fr in fold_results]
            fold_stats[f'{metric}_mean'] = np.mean(values)
            fold_stats[f'{metric}_std'] = np.std(values)
            fold_stats[f'{metric}_min'] = np.min(values)
            fold_stats[f'{metric}_max'] = np.max(values)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"LOSO Cross-Validation Summary ({n_subjects} folds)")
    print(f"{'='*60}")
    print(f"\nOverall Metrics (All Predictions Combined):")
    print_evaluation_summary(overall_metrics, "")
    
    print(f"Per-Fold Statistics (Mean ± Std):")
    for metric in metric_names:
        if f'{metric}_mean' in fold_stats:
            print(f"  {metric:20s}: {fold_stats[f'{metric}_mean']:.4f} ± {fold_stats[f'{metric}_std']:.4f} "
                  f"(min: {fold_stats[f'{metric}_min']:.4f}, max: {fold_stats[f'{metric}_max']:.4f})")
    print(f"{'='*60}\n")
    
    return {
        'fold_results': fold_results,
        'overall_metrics': overall_metrics,
        'fold_statistics': fold_stats,
        'all_predictions': all_predictions,
        'all_true_labels': all_true_labels,
        'all_probabilities': all_probabilities
    }


# Example usage
if __name__ == "__main__":
    # Replace with the directory you want to search
    start_dir = load_experimental_data()
    
    # Find and select directories
    selected_dirs = find_directories(
        start_dir,
        search_pattern="*RSVP_Calibration*",
        recursive=True,
        validate_bcipy=True
    )
    
    # Print selected directories
    print("Selected Directories:")
    for idx, path in enumerate(selected_dirs):
        print(f"  {idx + 1}. {path}")

    if len(selected_dirs) < 2:
        print("\nWarning: LOSO requires at least 2 subjects/directories.")
        print("Proceeding with standard training only.")

    # parameters_path = selected_dirs[0] + "/" + DEFAULT_PARAMETERS_FILENAME
    parameters_path = DEFAULT_PARAMETERS_PATH
    logging.info(f"Loading parameters from {parameters_path}")
    
    # Load parameters
    parameters = load_json_parameters(parameters_path, True)
    
    # Choose model type
    MODEL_TYPE = "PcaRdaKdeModel"  # Options: "PcaRdaKdeModel" or "EEGNet"
    print(f"\nModel Type: {MODEL_TYPE}")
    
    # Determine if external model format is needed
    external_model = (MODEL_TYPE == "EEGNet")
    
    # Load group trials with subject tracking
    print("\nLoading group trials...")
    trials, labels, device_spec, transform, subject_indices = load_group_trials(
        selected_dirs,
        parameters,
        external_model=external_model,
        return_per_subject=True
    )
    
    print(f"Loaded {trials.shape} and {len(labels)} labels from {len(selected_dirs)} subjects.")
    print(f"Class distribution: {np.sum(labels)} targets, {len(labels) - np.sum(labels)} non-targets")
    
    # Option 1: Leave-One-Subject-Out Cross-Validation
    if len(selected_dirs) >= 2:
        print("\n" + "="*60)
        print("Running Leave-One-Subject-Out (LOSO) Evaluation")
        print("="*60)
        
        loso_results = leave_one_subject_out_evaluation(
            trials=trials,
            labels=labels,
            subject_indices=subject_indices,
            device_spec=device_spec,
            default_transform=transform,
            model_type=MODEL_TYPE,
            save_models=False  # Set to True to save each fold's model
        )
        
        # Save LOSO results
        import json
        results_file = f"{GROUP_MODEL_NAME}_LOSO_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_results = {
                'fold_results': loso_results['fold_results'],
                'overall_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                   for k, v in loso_results['overall_metrics'].items()},
                'fold_statistics': {k: float(v) for k, v in loso_results['fold_statistics'].items()},
                'model_type': MODEL_TYPE,
                'n_subjects': len(selected_dirs),
                'n_trials': len(labels)
            }
            json.dump(serializable_results, f, indent=2)
        print(f"\nLOSO results saved to: {results_file}")
    
    # Option 2: Train a final group model on all data
    print("\n" + "="*60)
    print("Training Final Group Model on All Data")
    print("="*60)
    
    final_model = train_model(
        trials=trials,
        labels=labels,
        device_spec=device_spec,
        default_transform=transform,
        model_type=MODEL_TYPE
    )
    
    print(f"\nFinal model trained and saved.")
    print("\nNext steps:")
    print("  1. Review LOSO cross-validation results for generalization performance")
    print("  2. Use the final group model for new subject initialization")
    print("  3. Implement transfer learning/fine-tuning for individual subjects")
    
    breakpoint()  # Debugging breakpoint