from bcipy.signal.model.ar_offline import offline_analysis
# from bcipy.signal.model.offline_analysis import offline_analysis
from bcipy.helpers.load import load_experimental_data, load_json_parameters
from bcipy.config import DEFAULT_PARAMETER_FILENAME

from pathlib import Path
import mne


ARTIFACT_LABELLED_FILENAME = 'artifacts_raw.fif'

"""
Processing notes:

- The data is loaded into mne and then passed to offline_analysis. Done, this is significantly different than training on inquiry filtered data though. See auc_analysis.xlsx.
- *NEW* The data is loaded into mne, epochs are done with inquiry filtered data, then passed to offline_analysis.
- The data is loaded into mne with baseline interval and then passed to offline_analysis. Not done
- The data is loaded into mne with baseline interval applied, not passed in, and then passed to offline_analysis. Not done
- The data is loaded from mne AR and then passed to offline_analysis. Done
- the data is loaded from mne AR, bad epochs dropped and then passed to offline_analysis. *Done, but errors in training, meeting with @basak to discuss*
    ** Spoke with @basak and adjusted k-folds to 5 to account for potentially imbalanced classes. Re-running all sessions with lower k-folds to get baseline auc. This still doens't work for everyone.
- *NEW* The data is loaded into mne, epochs are done with inquiry filtered data, bad inquiries dropeed, then passed to offline_analysis.
- The data is loaded into mne, labelled automatically for artifacts, dropped, then passed to offline_analysis. Not done

"""

if __name__ == "__main__":
    """Process all sessions in a data folder and print the AUCs.
    
    This script is intended to be run from the command line with the following command:
        python offline_analysis_process_dataset.py
    """
    path = load_experimental_data()

    aucs = []
    ba = []
    for session in Path(path).iterdir():
        if session.is_dir():
            parameters = load_json_parameters(session / DEFAULT_PARAMETER_FILENAME, value_cast=True)
            # mne_data = mne.io.read_raw_fif(f'{session}/{ARTIFACT_LABELLED_FILENAME}')

            try:
                model, ba = offline_analysis(
                                # mne_data=mne_data,
                                data_folder=str(session.resolve()),
                                parameters=parameters,
                                drop_bad_epochs=False,
                                estimate_balanced_acc=True)
                aucs.append({
                    session.name: model.auc,
                    'BA': ba,
                    'model': model
                })
            except Exception as e:
                print(f"Error processing session {session}: \n {e}")
                import pdb; pdb.set_trace()
    
    import pdb; pdb.set_trace()
    print(aucs)
    import pdb; pdb.set_trace()