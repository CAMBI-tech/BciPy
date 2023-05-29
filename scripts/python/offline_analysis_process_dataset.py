from bcipy.signal.model.ar_offline import offline_analysis
# from bcipy.signal.model.offline_analysis import offline_analysis
from bcipy.helpers.load import load_experimental_data, load_json_parameters
from soa_classifier import crossvalidate_record, scores
from bcipy.config import DEFAULT_PARAMETER_FILENAME

from pathlib import Path
import mne
import pandas as pd
import numpy as np
import csv


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
- *NEW* The data is loaded into mne, epochs are done with inquiry filtered data, bad inquiries dropped, then passed to offline_analysis.
- The data is loaded into mne, labelled automatically for artifacts, dropped, then passed to offline_analysis. Not done

"""

if __name__ == "__main__":
    """Process all sessions in a data folder and print the AUCs.
    
    This script is intended to be run from the command line with the following command:
        python offline_analysis_process_dataset.py
    """
    path = load_experimental_data()

    results = {}
    dropped = {}
    for session in Path(path).iterdir():
        if session.is_dir():
            parameters = load_json_parameters(session / DEFAULT_PARAMETER_FILENAME, value_cast=True)
            mne_data = mne.io.read_raw_fif(f'{session}/{ARTIFACT_LABELLED_FILENAME}')

            try:
                # grab the data and labels
                results[session.name] = {}

                # comment out training to use the same data for all sessions. Use ar_offline.py to train for CF.
                data, labels, dropped = offline_analysis(
                                mne_data=mne_data,
                                data_folder=str(session.resolve()),
                                parameters=parameters,
                                drop_bad_epochs=False,
                                # baseline=(-.2, 0),
                                estimate_balanced_acc=True)
                # results[session.name]['ba'] = score
                # results[session.name]['auc'] = model.auc
                df = crossvalidate_record((data, labels))
                for name in scores:
                    results[session.name][name] = df[f'mean_test_{name}']
                dropped[session.name] = dropped
            except Exception as e:
                print(f"Error processing session {session}: \n {e}")
                breakpoint()
    print(results)
    file_name = 'scikit_cf_results_pca.csv'
    export = pd.DataFrame.from_dict(results).transpose()
    export.to_csv(file_name)

    breakpoint()
    # final = df.copy()
    # for session, values in results.items():
    #     for name, value in values.items():
    #         values = np.array(values)
    #         final[session][f'mean_test_{name}'] = values.mean(axis=0)
    #         final[session][f'std_test_{name}'] = values.std(axis=0)
    # final.sort_values('mean_test_mcc', ascending=False)
    
    # import pdb; pdb.set_trace()
    # final.to_csv('full_run_OF_final.csv')
    import pdb; pdb.set_trace()