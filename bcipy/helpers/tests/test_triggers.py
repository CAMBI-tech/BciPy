import ast
import unittest
from io import StringIO

import random
import bcipy.helpers.triggers as triggers


class TestTriggers(unittest.TestCase):
    """This is Test Case for Triggers."""

    def test_triggers_for_calibration(self):
        return

    def test_extract_from_calibration(self):
        trigger_seq = ['"[\'calibration_trigger\', 2.039073024992831]"', 'J',
                       '+', 'P', 'R', 'E', 'K', 'A', 'J',  'X', 'F', '<', 'S',
                       'E', '+', 'M', 'T', 'H', 'W', 'Y', 'V', 'E', 'L', '_',
                       'J']
        first_trg_time = 100
        trigger_interval = 10
        trigger_times = [first_trg_time + (i * trigger_interval)
                         for i in range(len(trigger_seq))]
        triggers_by_time = {k: v for k, v in zip(trigger_times, trigger_seq)}
        expected_targetness = ['calib', 'first_pres_target', 'fixation',
                               'nontarget', 'nontarget', 'nontarget',
                               'nontarget', 'nontarget', 'target', 'nontarget',
                               'nontarget', 'nontarget', 'nontarget',
                               'first_pres_target', 'fixation', 'nontarget',
                               'nontarget', 'nontarget', 'nontarget',
                               'nontarget', 'nontarget', 'target', 'nontarget',
                               'nontarget', 'nontarget']

        # Mock the raw_data file
        sep = '\r\n'
        meta = sep.join(['daq_type,TestStream', 'sample_rate,300'])
        header = 'timestamp,c1,c2,c3,TRG'

        data = []
        for i in range(1000):
            timestamp = i + 10.0
            channel_data = [str(random.uniform(-1000, 1000)) for _ in range(3)]
            trg = triggers_by_time.get(timestamp, triggers.NONE_VALUE)
            data.append(','.join([str(timestamp), *channel_data, trg]))

        content = sep.join([meta, header, *data])
        extracted = triggers.extract_from_calibration(StringIO(content),
                                                      seq_len=10)

        # Assertions
        self.assertEqual(len(trigger_seq), len(extracted))
        for seq_i in range(len(trigger_seq)):
            extracted_val, targetness, stamp = extracted[seq_i]
            trg = trigger_seq[seq_i]
            # The calibration has escaped chars, so it needs to be escaped
            # to test.
            expected_trg = ast.literal_eval(trg) if seq_i == 0 else trg
            self.assertEqual(expected_trg, extracted_val)
            self.assertEqual(expected_targetness[seq_i], targetness)
            self.assertEqual(trigger_times[seq_i], float(stamp))


if __name__ == '__main__':
    unittest.main()
