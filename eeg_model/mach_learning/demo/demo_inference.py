""" Inference demo """

import numpy as np
import string
from os import remove
from eeg_model.mach_learning.inference import inference
from helpers.load import read_data_csv, load_experimental_data
from eeg_model.mach_learning.trial_reshaper import trial_reshaper
from acquisition.sig_pro.sig_pro import sig_pro
from eeg_model.mach_learning.train_model import train_pca_rda_kde_model
from helpers.trigger_helpers import trigger_decoder


def _demo_inference():
    modes = ['calibration', 'copy_phrase', 'free_spell']
    sample_triggers = [\
    '< first_pres_target 0.670907955815',
    '+ fixation 2.65230878454',
    '< target 3.23335159523',
    'F nontarget 3.46778439358',
    'K nontarget 3.70046001566',
    'K nontarget 3.92969065203',
    'A nontarget 4.16515404402',
    'N nontarget 4.39758758984',
    'K nontarget 4.62848090783',
    'D nontarget 4.8619234586',
    'D nontarget 5.09161170403',
    'A nontarget 5.32642637414',
    'Y first_pres_target 6.49467007555',
    '+ fixation 8.47466339368',
    'Y target 9.05767254303',
    'R nontarget 9.29237042196',
    'G nontarget 9.52458454194',
    '< nontarget 9.75552882335',
    '< nontarget 9.98478034058',
    'B nontarget 10.2205293401',
    'D nontarget 10.4523640657',
    'P nontarget 10.6860699275',
    'O nontarget 10.9172955694',
    'N nontarget 11.1487296659',
    'A first_pres_target 12.2988197721',
    '+ fixation 14.2818938998',
    'A target 14.8640901118',
    'M nontarget 15.0989079671',
    'J nontarget 15.3305852016',
    '_ nontarget 15.562809939',
    'Z nontarget 15.7947462376',
    'C nontarget 16.0268616159',
    'O nontarget 16.2568418393',
    'L nontarget 16.4914501783',
    'R nontarget 16.722291825',
    'C nontarget 16.9548927715',
    'L first_pres_target 18.1060283357',
    '+ fixation 20.0890030139',
    'L target 20.6712063042',
    'J nontarget 20.9039095314',
    'M nontarget 21.1352675367',
    'S nontarget 21.3701048572',
    'U nontarget 21.6018058039',
    'P nontarget 21.8331666405',
    '< nontarget 22.065949852',
    'D nontarget 22.2980032956',
    'O nontarget 22.5301154888',
    'P nontarget 22.7622121098']

    data_folder = load_experimental_data()
    raw_data, _, _, _, fs_rate = read_data_csv(
        data_folder + '/rawdata.csv')
    ds_rate = 2  # Read from parameters file the down-sampling rate
    filtered_eeg = sig_pro(raw_data, fs=fs_rate, k=ds_rate) 

    for mode in modes:
        if mode == 'copy_phrase':
            sample_triggers = filter(lambda x: 'first_pres_target' 
                                     not in x,sample_triggers)

        if mode == 'free_spell':
            sample_triggers = map(lambda x: x.replace('fixation', '').replace('target', '').replace('non', ''),
                              filter(lambda x: 'first_pres_target' not in x,sample_triggers))

        with open('triggers.txt', 'w+') as text_file:
            for line in sample_triggers:
                text_file.write(line + '\n')
        
        # Load trigger file
        trigger_data = trigger_decoder(trigger_loc='triggers.txt', mode=mode)

        # Remove the trigger.txt file that has been used.
        remove('triggers.txt')

        trial_target_info = trigger_data[1]
        timing_info = trigger_data[2]
        
        # Get data and labels
        reshaped_trials, labels, _, _ = trial_reshaper(
            trial_target_info, timing_info, filtered_eeg, fs_rate, ds_rate, mode)

        # Determine on number of folds based on the data!
        k_folds = 4
        model = train_pca_rda_kde_model(reshaped_trials, labels, k_folds=k_folds)
        
        # This creates a dictionary of letters with the keys being letters
        # and values being integers:
        letters = {k: v for v, k in enumerate(string.ascii_uppercase, 1)}
        # Letters in the alphabet, each represented as an integer:
        alphabet = np.array(list(letters.values()))
        targets = "This is inference demo"
        targets = targets.replace(" ", "")
        # This returns an array of letters in the target sentence:
        targets = np.array(list(list(targets)))
        lik_r = inference(alphabet, reshaped_trials, labels, model)
        
        print "Inference Flows!" 
        print "Log Likelihood ratios mapped to the flashed letters for \
               calibration mode: " + mode + " is:"
        print lik_r
        return 0


def main():
    _demo_inference()

    return 0


if __name__ == "__main__":
    main()
