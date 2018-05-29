import numpy as np
from signal_processing.sig_pro import sig_pro
from signal_model.mach_learning.trial_reshaper import trial_reshaper
import time
from signal_model.inference import inference
from bci_tasks.main_frame import EvidenceFusion, DecisionMaker
from signal_model.mach_learning.train_model import train_pca_rda_kde_model
from helpers.bci_task_related import alphabet

# TODO: These are shared parameters for multiple functions
# and I have no idea how to put them into correct place
channel_map = [0] + [1] * 16 + [0, 0, 1, 1, 0, 1, 1, 1, 0]
dim_x = 5
num_ch = len(channel_map)

# Make up some distributions that data come from
mean_pos = .5
var_pos = .5
mean_neg = 0
var_neg = .5


# TODO: Distributions are hardcoded!!
def dummy_trig_dat_generator(truth, state, stimuli):
    """ Dummy trigger and data generation, will be replaced with daq and
        display in the code. Appends sampled data to an array returns it as
        the eeg evidence.
        Args:
            truth(str): phrase in the copy phrase task
            state(str): state of the decision maker
            stimuli(str): symbols in the current stimuli
        Return:
            eeg(ndarray[float]): C x L eeg data where C is number of
                channels and L is the signal length
            trigger(list[tuple(str,float)]): triggers e.g. ('A', 1)
                as letter and flash time for the letter
            target_info(list[str]): target information about the stimuli
            """
    # Remove fixation
    if truth[0:len(state)] == state:
        tar = truth[len(state)]
    else:
        tar = '<'

    if any(tar in i for i in stimuli):
        target_info = ['Non_Target'] * len(stimuli)
        idx = stimuli.index(tar)

        tmp = mean_neg + var_neg * np.random.randn(num_ch, len(stimuli), dim_x)
        tmp[:, idx, :] = mean_pos + var_pos * np.random.randn(num_ch, dim_x)
        target_info[idx] = 'Target'
    else:
        target_info = ['Non_Target'] * len(stimuli)
        tmp = mean_neg + var_neg * np.random.randn(num_ch, len(stimuli), dim_x)

    tmp = [tmp[:, i, :] for i in range(tmp.shape[1])]
    eeg = np.concatenate(tmp, axis=1)

    time = np.array(list(range(len(stimuli)))) / 2.
    trigger = [(stimuli[i], time[i]) for i in range(len(stimuli))]

    return eeg, trigger, target_info


class CopyPhraseWrapper(object):
    """Basic copy phrase task duty cycle wrapper.

    Given the phrases once operate() is called performs the task.
    Attr:
        conjugator(EvidenceFusion): fuses evidences in the task
        decision_maker(DecisionMaker): mastermind of the task
        alp(list[str]): symbol set of the task
        model(pipeline): model trained using a calibration session of the
            same user.
        fs(int): sampling frequency
        k(int): down sampling rate
        mode(str): mode of thet task (should be copy phrase)
        d(binary): decision flag
        sti(list(tuple)): stimuli for the display
        task_list(list[tuple(str,str)]): list[(phrases, initial_states)] for
            the copy phrase task
    """

    def __init__(self, signal_model, fs, k, alp, evidence_names=['LM', 'ERP'],
                 task_list=[('I_LOVE_COOKIES', 'I_LOVE_')], lmodel=None,
                 is_txt_sti=True):

        self.conjugator = EvidenceFusion(evidence_names, len_dist=len(alp))
        self.decision_maker = DecisionMaker(state=task_list[0][1],
                                            alphabet=alp,
                                            is_txt_sti=is_txt_sti)
        self.alp = alp

        self.signal_model = signal_model
        self.fs = fs
        self.k = k

        self.mode = 'copy_phrase'
        self.task_list = task_list
        self.lmodel = lmodel

    def do_sequence(self):
        """Display symbols and collect evidence."""

        self.decision_maker.displayed_state
        self.sti

        raw_dat, triggers, target_info = 0, 0, 0

        return raw_dat, triggers, target_info

    def evaluate_sequence(self, raw_dat, triggers, target_info):
        """Once data is collected, infers meaning from the data.

        Args:
            raw_dat(ndarray[float]): C x L eeg data where C is number of
                channels and L is the signal length
            triggers(list[tuple(str,float)]): triggers e.g. ('A', 1)
                as letter and flash time for the letter
            target_info(list[str]): target information about the stimuli
        """

        try:
            # Send the raw data to signal processing / in demo mode do not use sig_pro
            # dat = sig_pro(raw_dat, fs=self.fs, k=self.k)
            dat = raw_dat

            # TODO: if it is a fixation remove it don't hardcode it as if you did
            letters = [triggers[i][0] for i in range(0, len(triggers))]
            time = [triggers[i][1] for i in range(0, len(triggers))]

            # Raise an error if the stimuli includes unexpected terms
            if not set(letters).issubset(set(self.alp+['+'])):
                raise 'stimuli include letters not in {alphabet, {'+'}}'

            # Remove information in any trigger related with the fixation
            del target_info[letters.index('+')]
            del time[letters.index('+')]
            del letters[letters.index('+')]

            x, y, _, _ = trial_reshaper(target_info, time, dat, fs=self.fs,
                                        k=self.k, mode=self.mode,
                                        channel_map=channel_map)

            lik_r = inference(x, letters, self.signal_model, self.alp)
            prob = self.conjugator.update_and_fuse({'ERP': lik_r})
            decision, arg = self.decision_maker.decide(prob)

            if 'stimuli' in arg:
                sti = arg['stimuli']
            else:
                sti = None

        except Exception as e:

            print("Error in evaluate_sequence: %s" % (e))
            raise e

        return decision, sti

    def initialize_epoch(self):
        """If a decision is made initializes the next epoch."""

        try:
            # First, reset the history for this new epoch
            self.conjugator.reset_history()

            # If there is no language model specified, mock the LM prior
            if not self.lmodel:
                # get probabilites from language model
                prior = np.ones(len(self.alp))
                prior /= np.sum(prior)

            # Else, let's query the lmodel for priors
            else:
                # Get the displayed state, and do a list comprehension to place
                # in a form the LM recognizes
                update = [letter
                          if not letter == '_'
                          else ' '
                          for letter in self.decision_maker.displayed_state]

                # update the lmodel and get back the priors
                lm_prior = self.lmodel.state_update(update)

                # hack: Append it with a backspace
                lm_prior['prior'].append(['<', 0])

                # construct the priors as needed for evidence fusion
                prior = [float(pr_letter[1])
                         for alp_letter in self.alp
                         for pr_letter in lm_prior['prior']
                         if alp_letter == pr_letter[0]
                         ]

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

    def operate(self):
        """Main function of the task. Once called, performs the task."""

        for task in self.task_list:
            task_final = task[0]
            task_state = task[1]

            self.decision_maker.update(task_state)
            loop_counter = 0
            print('task:{}'.format(task_final))
            d = 1

            # TODO: Get stopping condition as parameter
            while (self.decision_maker.displayed_state != task_final and
                   loop_counter < 50):
                if d:
                    d, sti = self.initialize_epoch()
                    if sti:
                        sti = sti[0][0]
                else:
                    # raw_dat, triggers, target_info = self.do_sequence()
                    raw_dat, triggers, target_info = dummy_trig_dat_generator(
                        task_final, self.decision_maker.displayed_state,
                        sti)

                    d, sti = self.evaluate_sequence(raw_dat, triggers,
                                                    target_info)
                    if sti:
                        sti = sti[0][0]

                # TODO: sleep for demo purposes. Remove it afterwards
                time.sleep(.3)
                print('\rstate:{}'.format(self.decision_maker.state)),
            print('')


def demo_copy_phrase_wrapper():
    # We need to train a dummy model
    num_x_p = 100
    num_x_n = 900

    x_p = mean_pos + var_pos * np.random.randn(num_ch, num_x_p, dim_x)
    x_n = mean_neg + var_neg * np.random.randn(num_ch, num_x_n, dim_x)
    y_p = [1] * num_x_p
    y_n = [0] * num_x_n

    train_x = np.concatenate((x_p, x_n), 1)
    train_y = np.concatenate((y_p, y_n), 0)
    permutation = np.random.permutation(train_x.shape[1])
    train_x = train_x[:, permutation, :]
    train_y = train_y[permutation]

    train_x = train_x[list(np.where(np.asarray(channel_map) == 1)[0]), :, :]

    k_folds = 10
    model = train_pca_rda_kde_model(train_x, train_y, k_folds=k_folds)

    # Define task and operate
    task_list = [('I_LOVE_COOKIES', 'I_LOVE_'),
                 ('THIS_IS_A_DEMO', 'THIS_IS_A_')]

    task = CopyPhraseWrapper(model, fs=dim_x * 2, k=1, alp=alphabet(),
                             task_list=task_list)
    task.operate()


if __name__ == "__main__":
    demo_copy_phrase_wrapper()
