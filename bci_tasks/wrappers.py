import numpy as np
from eeg_model.mach_learning.trial_reshaper import trial_reshaper
import time
from acquisition.sig_pro.sig_pro import sig_pro
from eeg_model.inference import inference
from bci_tasks.main_frame import EvidenceFusion, DecisionMaker
from eeg_model.mach_learning.train_model import train_pca_rda_kde_model

# TODO: These are shared parameters for multiple functions
# and I have no idea how to put them into correct place
dim_x = 50
num_ch = 16

# Make up some distributions that data come from
mean_pos = .2
var_pos = .5
mean_neg = 0
var_neg = .5

# Symbol set
alp = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
       'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', '<', '_']


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
    stimuli.pop(0)
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
    """ Basic copy phrase task duty cycle wrapper. Given the phrases once
    operate() is called performs the task.
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

    def __init__(self, model, fs, k, alp, evidence_names=['LM', 'ERP'],
                 task_list=[('I_LOVE_COOKIES', 'I_LOVE_')]):
        self.conjugator = EvidenceFusion(evidence_names, len_dist=len(alp))
        self.decision_maker = DecisionMaker(state=task_list[0][1],
                                            alphabet=alp)
        self.alp = alp

        self.model = model
        self.fs = fs
        self.k = k

        self.mode = 'copy_phrase'
        self.d = 1
        self.sti = None

        self.task_list = task_list

    def do_sequence(self):
        """ Display symbols and collect evidence """
        # TODO: display parameters
        self.decision_maker.displayed_state
        self.sti

        # TODO: get data triggers and target info
        raw_dat, triggers, target_info = 0, 0, 0

        return raw_dat, triggers, target_info

    def evaluate_sequence(self, raw_dat, triggers, target_info):
        """ Once data is collected, infers meaning from the data.
            Args:
                raw_dat(ndarray[float]): C x L eeg data where C is number of
                    channels and L is the signal length
                triggers(list[tuple(str,float)]): triggers e.g. ('A', 1)
                    as letter and flash time for the letter
                target_info(list[str]): target information about the stimuli
                """
        # dat = sig_pro(raw_dat, fs=self.fs, k=self.k)
        dat = raw_dat
        letters = [triggers[i][0] for i in range(len(triggers))]
        time = [triggers[i][1] for i in range(len(triggers))]

        x, y, _, _ = trial_reshaper(target_info, time, dat, fs=self.fs,
                                    k=self.k, mode=self.mode)
        lik_r = inference(x, letters, self.model, self.alp)
        p = self.conjugator.update_and_fuse({'ERP': lik_r})
        self.d, arg = self.decision_maker.decide(p)

        if 'stimuli' in arg:
            self.sti = arg['stimuli'][0][0]

    def initialize_epoch(self):
        """ If a decision is made initializes the next epoch """
        self.conjugator.reset_history()

        # TODO: update language model with
        self.decision_maker.displayed_state
        # get probabilites from language model
        prior = np.ones(len(alp))
        prior /= np.sum(prior)

        p = self.conjugator.update_and_fuse({'LM': prior})
        self.d, arg = self.decision_maker.decide(p)
        self.sti = arg['stimuli'][0][0]

    def operate(self):
        """ Main function of the task. Once called, performs the task """
        for task in self.task_list:
            task_final = task[0]
            task_state = task[1]

            self.decision_maker.update(task_state)
            loop_counter = 0
            print('task:{}'.format(task_final))

            # TODO: Get stopping condition as parameter
            while (self.decision_maker.displayed_state != task_final and
                           loop_counter < 50):
                if self.d:
                    self.initialize_epoch()
                else:
                    # raw_dat, triggers, target_info = self.do_sequence()
                    raw_dat, triggers, target_info = dummy_trig_dat_generator(
                        task_final, self.decision_maker.displayed_state,
                        self.sti)
                    self.evaluate_sequence(raw_dat, triggers, target_info)

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
    train_y = np.concatenate(np.asarray([y_p, y_n]), 0)
    permutation = np.random.permutation(train_x.shape[1])
    train_x = train_x[:, permutation, :]
    train_y = train_y[permutation]

    k_folds = 10
    model = train_pca_rda_kde_model(train_x, train_y, k_folds=k_folds)

    # Define task and operate
    task_list = [('I_LOVE_COOKIES', 'I_LOVE_'),
                 ('THIS_IS_A_DEMO', 'THIS_IS_A_')]

    task = CopyPhraseWrapper(model, fs=dim_x * 2, k=1, alp=alp,
                             task_list=task_list)
    task.operate()


if __name__ == "__main__":
    demo_copy_phrase_wrapper()
