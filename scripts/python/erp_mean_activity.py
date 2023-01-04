import mne
mne.set_log_level('WARNING')
from pathlib import Path
from bcipy.helpers.load import (
    load_experimental_data,
)

if __name__ == '__main__':
    write_output = False
    target_filename = 'target-epo.fif'
    nontarget_filename = 'nontarget-epo.fif'
    raw_data_filename = 'artifacts_raw.fif'

    # process the target data first and use those as the first place label for the nontarget data

    path = load_experimental_data()
    
    for session in Path(path).iterdir():
        try:
            import pdb; pdb.set_trace()
            # mne_data = mne.io.read_raw_fif(f'{session}/{raw_data_filename}', preload=True)
            target = mne.read_epochs(f'{session}/{target_filename}', preload=True)
            nontarget = mne.read_epochs(f'{session}/{nontarget_filename}', preload=True)

            if write_output:
                # TODO: write the output to a file
                with open(f'{session}/n2_p3_meanactivity.txt', 'w') as f:
                    f.write(f'Label, Target, Nontarget \n')
                    f.write(f'N2, {n2_target}, {n2_nontarget} \n')
                    f.write(f'P3, {p3_target}, {p3_nontarget} \n')
            
        except Exception as e:
            print(f'Could not load epochs for session {session}: [{e}]')
            continue