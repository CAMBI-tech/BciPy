import numpy as np
from bcipy.helpers.bci_task_related import trial_reshaper
from bcipy.signal.model.inference import inference
from bcipy.signal.processing.sig_pro import sig_pro
from bcipy.tasks.main_frame import EvidenceFusion, DecisionMaker
from bcipy.helpers.acquisition_related import analysis_channels
from bcipy.helpers.lang_model_related import norm_domain

class CopyPhraseWrapper(object):
    """Basic copy phrase task duty cycle wrapper.

    Given the phrases once operate() is called performs the task.
    Attr:
        min_num_seq: The minimum number of sequences to be displayed
        max_num_seq: The maximum number of sequences to be displayed
        model(pipeline): model trained using a calibration session of the
            same user.
        fs(int): sampling frequency
        k(int): down sampling rate
        alp(list[str]): symbol set of the task
        task_list(list[tuple(str,str)]): list[(phrases, initial_states)] for
            the copy phrase task
        is_txt_sti: Whether or not the stimuli are text objects
        conjugator(EvidenceFusion): fuses evidences in the task
        decision_maker(DecisionMaker): mastermind of the task
        mode(str): mode of thet task (should be copy phrase)
        d(binary): decision flag
        sti(list(tuple)): stimuli for the display
    """
    def __init__(self, min_num_seq, max_num_seq, signal_model=None, fs=300, k=2,
                 alp=None, evidence_names=['LM', 'ERP'],
                 task_list=[('I_LOVE_COOKIES', 'I_LOVE_')], lmodel=None,
                 is_txt_sti=True, device_name='LSL', device_channels=None,
                 stimuli_timing=[1, .2]):

        self.conjugator = EvidenceFusion(evidence_names, len_dist=len(alp))
        self.decision_maker = DecisionMaker(min_num_seq, max_num_seq,
                                            state=task_list[0][1],
                                            alphabet=alp,
                                            is_txt_sti=is_txt_sti,
                                            stimuli_timing=stimuli_timing)
        self.alp = alp

        self.signal_model = signal_model
        self.fs = fs
        self.k = k

        self.mode = 'copy_phrase'
        self.task_list = task_list
        self.lmodel = lmodel
        self.channel_map = analysis_channels(device_channels, device_name)

    def evaluate_sequence(self, raw_dat, triggers, target_info, window_length):
        """Once data is collected, infers meaning from the data.

        Args:
            raw_dat(ndarray[float]): C x L eeg data where C is number of
                channels and L is the signal length
            triggers(list[tuple(str,float)]): triggers e.g. ('A', 1)
                as letter and flash time for the letter
            target_info(list[str]): target information about the stimuli
            window_length(int): The length of the time between stimuli presentation
        """
        # Send the raw data to signal processing / in demo mode do not use sig_pro
        dat = sig_pro(raw_dat, fs=self.fs, k=self.k)

        # TODO: if it is a fixation remove it don't hardcode it as if you did
        letters = [triggers[i][0] for i in range(0, len(triggers))]
        time = [triggers[i][1] for i in range(0, len(triggers))]

        # Raise an error if the stimuli includes unexpected terms
        if not set(letters).issubset(set(self.alp+['+']+['PLUS']+['calibration_trigger'])):
            raise Exception('unexpected letters received in copy phrase')

        # Remove information in any trigger related with the fixation
        if '+' in letters:
            del_letter = '+'
        elif 'PLUS' in letters:
            del_letter = 'PLUS'
        else:
            raise Exception('could not find target + sign in letters')

        if 'calibration_trigger' in letters:
            del target_info[letters.index('calibration_trigger')]
            del time[letters.index('calibration_trigger')]
            del letters[letters.index('calibration_trigger')]

        del target_info[letters.index(del_letter)]
        del time[letters.index(del_letter)]
        del letters[letters.index(del_letter)]

        x, _, _, _ = trial_reshaper(target_info, time, dat, fs=self.fs,
                                    k=self.k, mode=self.mode,
                                    channel_map=self.channel_map,
                                    trial_length=window_length)

        lik_r = inference(x, letters, self.signal_model, self.alp)
        prob = self.conjugator.update_and_fuse({'ERP': lik_r})
        decision, arg = self.decision_maker.decide(prob)

        if 'stimuli' in arg:
            sti = arg['stimuli']
        else:
            sti = None

        return decision, sti

    def initialize_epoch(self):
        """If a decision is made initializes the next epoch."""

        try:
            # First, reset the history for this new epoch
            self.conjugator.reset_history()

            # If there is no language model specified, mock the LM prior
            # TODO: is the probability domain correct? ERP evidence is in
            # the log domain; LM by default returns negative log domain.
            if not self.lmodel:
                # mock probabilities to be equally likely for all letters.
                n_letters = len(self.alp)
                prior = np.full(n_letters, 1/n_letters)

            # Else, let's query the lmodel for priors
            else:
                # Get the displayed state
                # TODO: for oclm this should be a list of (sym, prob)
                update = self.decision_maker.displayed_state

                # update the lmodel and get back the priors
                lm_prior = self.lmodel.state_update(update)

                # normalize to probability domain
                lm_letter_prior = norm_domain(lm_prior['letter'])

                # hack: Append it with a backspace
                if not '<' in dict(lm_letter_prior):
                    lm_letter_prior.append(('<', 0.0))

                # convert to format needed for evidence fusion;
                # probability value only in alphabet order.
                prior = [prior_prob
                         for alp_letter in self.alp
                         for prior_sym, prior_prob in lm_letter_prior
                         if alp_letter == prior_sym]

            # Try fusing the lmodel evidence
            try:
                p = self.conjugator.update_and_fuse({'LM': np.array(prior)})
            except Exception as e:
                print("Error updating language model!")
                raise e

            # Get decision maker to give us back some decisions and stimuli
            d, arg = self.decision_maker.decide(p)
            sti = arg['stimuli']

        except Exception as e:
            print("Error in initialize_epoch: %s" % (e))
            raise e

        return d, sti
