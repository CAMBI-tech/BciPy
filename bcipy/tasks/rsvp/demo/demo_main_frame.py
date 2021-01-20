from bcipy.tasks.rsvp.main_frame import EvidenceFusion, DecisionMaker
import numpy as np


def _demo_fusion():
    len_alp = 4
    evidence_names = ['LM', 'ERP', 'FRP']
    num_series = 4
    num_inquiries = 10

    conjugator = EvidenceFusion(evidence_names, len_dist=len_alp)

    print('Random Series!')
    for idx_ep in range(num_series):
        prior = np.abs(np.random.randn(len_alp))
        prior = prior / np.sum(prior)
        conjugator.update_and_fuse({'LM': prior})
        for idx in range(num_inquiries):
            # Generate random inquiries
            evidence_erp = 10 * np.abs(np.random.randn(len_alp))
            evidence_frp = 10 * np.abs(np.random.randn(len_alp))
            conjugator.update_and_fuse(
                {'ERP': evidence_erp, 'FRP': evidence_frp})
        print('Series: {}'.format(idx_ep))
        print(conjugator.evidence_history['ERP'])
        print(conjugator.evidence_history['FRP'])
        print(conjugator.evidence_history['LM'])
        print('Posterior:{}'.format(conjugator.likelihood))

        # Reset the conjugator before starting a new series for clear history
        conjugator.reset_history()


def _demo_decision_maker():
    alp = ['T', 'H', 'I', 'S', 'I', 'S', 'D', 'E', 'M', 'O']
    len_alp = len(alp)
    evidence_names = ['LM', 'ERP', 'FRP']
    num_series = 10

    conjugator = EvidenceFusion(evidence_names, len_dist=len_alp)
    decision_maker = DecisionMaker(
        min_num_inq=1,
        max_num_inq=10,
        state='',
        alphabet=alp)

    for idx_series in range(num_series):

        while True:
            # Generate random inquiries
            evidence_erp = np.abs(np.random.randn(len_alp))
            evidence_erp[idx_series] += 1
            evidence_frp = np.abs(np.random.randn(len_alp))
            evidence_frp[idx_series] += 3

            p = conjugator.update_and_fuse(
                {'ERP': evidence_erp, 'FRP': evidence_frp})

            d, arg = decision_maker.decide(p)
            if d:
                break
        # Reset the conjugator before starting a new series for clear history
        conjugator.reset_history()

    print('State:{}'.format(decision_maker.state))
    print('Displayed State: {}'.format(decision_maker.displayed_state))


if __name__ == '__main__':
    _demo_decision_maker()
    _demo_fusion()
