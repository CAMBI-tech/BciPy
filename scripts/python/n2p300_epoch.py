"""
Analysis Notes
--------------

This script processes a folder of session data, looping over each session and processing the data into an average of epochs by poisition accross subjects and trials.

***Done outside of this script***
--------------------------------
Filtered: Yes (default) 1-20 Hz, 2nd order butterworth, 60 Hz notch filter
Artifact Rejection: Semi-automatic (no more than 50% bad epochs per condition)

***Done in this script***
-------------------------
Baseline: -200ms to 0ms
Epochs: 0ms to 800ms; Pz or pooled signal (average of P07, P08, Pz, Cz)
"""
import mne
mne.set_log_level('WARNING')
from pathlib import Path
from typing import Tuple, List
from bcipy.config import DEFAULT_PARAMETER_FILENAME, TRIGGER_FILENAME, RAW_DATA_FILENAME
from bcipy.helpers.load import (
    load_experimental_data,
    load_json_parameters,
)
from bcipy.helpers.visualization import visualize_evokeds, visualize_joint_average
from bcipy.helpers.stimuli import mne_epochs
from bcipy.helpers.triggers import TriggerType, trigger_decoder

ARTIFACT_LABELLED_FILENAME = 'artifacts_raw.fif'


def epoch_prelabelled_data(
        path: Path,
        baseline: Tuple[float]=(-0.2, 0.0),
        percent_bad: float=50.0) -> Tuple[mne.Epochs, list, mne.io.RawArray]:
    """Epoch prelabelled data.
    
    Parameters
    ----------
    baseline : tuple
        The baseline to use for the epochs.
    percent_bad : float
        The percentage of bad epochs to allow before rejecting a session from the analysis.
    
    Returns
    -------
    epochs : mne.Epochs 
        The target/nontarget epochs for the session.
    positions : list
        positions or keys for the epochs.
    mne_data : mne.io.RawArray

    """
    parameters = load_json_parameters(f'{path}/{DEFAULT_PARAMETER_FILENAME}', value_cast=True)

    # extract all relevant parameters
    poststim_length = parameters.get("trial_length")
    # prestim_length = parameters.get("prestim_length") # used for online filtering
    # trials_per_inquiry = parameters.get("stim_length")
    trials = parameters.get("stim_number")

    static_offset = parameters.get("static_trigger_offset")

    # load mne data
    mne_data = mne.io.read_raw_fif(f'{path}/{ARTIFACT_LABELLED_FILENAME}', preload=True)

    # process triggers.txt files
    trigger_targetness, trigger_timing, _ = trigger_decoder(
        offset=static_offset,
        trigger_path=f"{path}/{TRIGGER_FILENAME}",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
    )

    # convert the labels from string to 0/1
    labels = [0 if label == 'nontarget' else 1 for label in trigger_targetness]
    epochs = mne_epochs(
        mne_data,
        trigger_timing,
        poststim_length,
        labels,
        baseline=baseline,
        reject_by_annotation=True,
        channels=None)
    epochs.drop_bad() # TODO: verify if this is necessary

    # check if we have enough good epochs to include this session in the analysis
    clases = [1, 2]
    permitted_loss = trials * (1 - (percent_bad / 100))
    for pos in clases:
        epoch_pos = epochs[f'{pos}']
        print(f'Class {pos} has {len(epoch_pos)} epochs')
        if len(epoch_pos) < permitted_loss:
            print('Too many epochs rejected. This session will be excluded from analysis.')
            return None

    return epochs, clases, mne_data


def grand_average(epochs: mne.Epochs, positions: List[int], labels: List[str]=['NonTarget', 'Target']) -> mne.EvokedArray:
    """Compute the grand average of epochs by class.
    
    Parameters
    ----------
    epochs : mne.Epochs
        All epochs for the session with corresponding keys listed in positions.
    positions : list
        Position keys for the epochs.
    labels : list
        Labels for each of the classes.

    Returns
    -------
    evokeds : mne.EvokedArray
    """
    assert len(positions) == len(labels), (
        f'length of labels ({len(labels)}) should be same as length of position, ({len(positions)})'
    )
    concat_epochs = []
    for position in positions:
        concat_epochs.append(
            mne.concatenate_epochs([epoch[f'{position}'] for epoch in epochs])
        )

    visualize_evokeds(concat_epochs, show=True)
    visualize_joint_average(concat_epochs, labels, show=True)
    return concat_epochs


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--path',
        help='Path to the directory with raw_data to be converted',
        required=False)
    args = parser.parse_args()

    path = args.path
    if not path:
        path = load_experimental_data()
    
    all_epochs = []
    all_excluded = []
    positions = None
    for session in Path(path).iterdir():
        if session.is_dir():
            print(f'Processing {session}')
            try: 
                resp = epoch_prelabelled_data(session)
                if resp:
                    epochs, positions, _ = resp
                    all_epochs.append(epochs)
                    # epochs.save(f'{session}/target_nontarget_epochs.fif', overwrite=True)
                else:
                    all_excluded.append(session)
                    print(f'Excluding {session}')
            except Exception as e:
                print(f'Error processing {session}: {e}')
                all_excluded.append(session)
                print(f'Excluding {session}')
        
        # debugger to ensure we should continue processing
        # import pdb; pdb.set_trace()

    # debugger to catch everything at the end and evaluate results
    import pdb; pdb.set_trace()
    # Grand averaging TODO: figure this out!
    # grand_average = mne.grand_average(all_epochs)
    # grand_average(all_epochs, positions)


    import pdb; pdb.set_trace()