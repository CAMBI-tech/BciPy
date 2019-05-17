"""Tests for Copy Phrase EvidenceFusion"""

import unittest
import numpy as np
import bcipy.tasks.rsvp.main_frame as mf
from bcipy.helpers.load import load_json_parameters


class TestEvidenceFusion(unittest.TestCase):
    """Tests for EvidenceFusion"""

    def test_fusion(self):
        len_alp = 4
        evidence_names = ['LM', 'ERP', 'FRP']
        num_sequences = 10

        conjugator = mf.EvidenceFusion(evidence_names, len_dist=len_alp)

        # generated random sequences
        erp_evidence = [
            np.array([1.46762823, 0.40728661, 9.48721375, 11.47963171]),
            np.array([6.18696887, 8.66360132, 4.10906719, 10.40379058]),
            np.array([14.842917, 7.95145995, 17.68779128, 10.94195076]),
            np.array([2.1032388, 14.56945629, 9.9442332, 8.59807331]),
            np.array([17.2194632, 7.49341414, 10.7793301, 8.42595077]),
            np.array([12.78049307, 3.22652363, 0.55427707, 14.64100114]),
            np.array([7.23923751, 26.61319581, 7.72974217, 1.82107856]),
            np.array([6.35755791, 1.40185201, 5.94498732, 2.88180149]),
            np.array([13.72248648, 8.91011389, 13.76731832, 15.66952482]),
            np.array([16.64688501, 8.2827469, 1.75102074, 9.12089907])
        ]
        frp_evidence = [
            np.array([2.59452579, 18.37016033, 0.99017855, 7.75622695]),
            np.array([14.43263341, 10.97966831, 0.08324027, 7.39021412]),
            np.array([13.17879992, 6.62931003, 6.27445546, 7.72110125]),
            np.array([7.78663088, 17.96760446, 2.51192754, 6.26679419]),
            np.array([8.63589394, 10.97667028, 8.02164873, 26.98926272]),
            np.array([7.49884707, 12.56370517, 7.77508668, 1.45465084]),
            np.array([5.94854662, 5.13292461, 7.74513707, 3.78547598]),
            np.array([7.19402239, 0.32145887, 9.15348207, 1.86521141]),
            np.array([6.61159441, 0.14524912, 8.85456089, 2.65464896]),
            np.array([5.44631832, 3.94499887, 25.16119557, 9.28041577])
        ]
        lm_evidence = np.array([0.3198295, 0.24571754, 0.06543074, 0.36902222])

        expected_posterior = [
            9.16884720e-01, 2.38754536e-04, 4.33163904e-05, 8.28332090e-02
        ]

        # Single epoch

        # initialize with language model priors
        conjugator.update_and_fuse({'LM': lm_evidence})

        for idx in range(num_sequences):
            conjugator.update_and_fuse({
                'ERP': erp_evidence[idx],
                'FRP': frp_evidence[idx]
            })

        self.assertEqual(1, len(conjugator.evidence_history['LM']))
        for i, val in enumerate(conjugator.evidence_history['LM'][0]):
            self.assertEqual(lm_evidence[i], val)

        for i, val in enumerate(conjugator.evidence_history['ERP']):
            self.assertSequenceEqual(list(erp_evidence[i]), list(val))

        for i, val in enumerate(conjugator.evidence_history['FRP']):
            self.assertSequenceEqual(list(frp_evidence[i]), list(val))

        for i, val in enumerate(conjugator.likelihood):
            self.assertAlmostEqual(expected_posterior[i], val)
