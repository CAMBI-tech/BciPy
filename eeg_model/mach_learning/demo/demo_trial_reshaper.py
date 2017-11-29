import numpy as np
from eeg_model.mach_learning.trial_reshaper import trial_reshaper
from helpers.triggers import trigger_decoder
from os import remove

# A 3 channel dummy input
inp = np.array([range(4000)] * 3)
# Sampling frequency
fs = 256
# Downsampling ratio
k = 2

# Uncomment the calibration mode you want to test trial reshaper for below:
# demo_mode = 'calibration'
# demo_mode = 'copy_phrase'
demo_mode = 'free_spell'

# Create a mock triggers.txt according to demo_mode
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

if demo_mode == 'copy_phrase':
    sample_triggers = filter(lambda x: 'first_pres_target' not in x,sample_triggers)

if demo_mode == 'free_spell':
    sample_triggers = map(lambda x: x.replace('fixation', '').replace('target', '').replace('non', ''),filter(lambda x: 'first_pres_target' not in x,sample_triggers))

with open('triggers.txt', 'w+') as text_file:
    for line in sample_triggers:
        text_file.write(line + '\n')

# Load trigger file
trigger_data = trigger_decoder(trigger_loc='triggers.txt', mode=demo_mode)

# Remove the trigger.txt file that has been used.
remove('triggers.txt')

# reshape function is applied to dummy data with given trigger file
arg = trial_reshaper(trial_target_info=trigger_data[1],
                     timing_info=trigger_data[2], filtered_eeg=inp, fs=256, k=2, mode=demo_mode)

# Print results.
print 'Reshaped trials:\n', arg[0], '\nLabels:', arg[1], '\nTotal number of sequences:', \
    arg[2], '\nTrial number in each sequence:', arg[3]
