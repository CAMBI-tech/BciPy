import unittest
from io import StringIO
from typing import List, Tuple, Dict
import random
from bcipy.helpers.triggers import NONE_VALUE, CopyPhraseClassifier, \
    extract_from_copy_phrase, extract_from_calibration, \
    write_trigger_file_from_lsl_calibration, \
    write_trigger_file_from_lsl_copy_phrase


def sample_raw_data(trigger_seq: List[Tuple[str, str]] =[],
                    first_trg_time: int=100,
                    trigger_interval: int=10) -> Tuple[str, List[float]]:
    """Helper function for creating mock data that looks like the raw_data.csv
    output. Adds trigger data to the TRG column at the specified interval.

    Parameters:
    -----------
        trigger_seq: list of tuples with the stimulus, targetness.
        first_trg_time: first time in the data where a trigger should appear.
        trigger_interval: set interval at which subsequent triggers should be
                    displayed
    Returns:
    --------
        content: str, trigger_times: list(float)
    """
    # Set up the trigger times
    trigger_times = []
    triggers_by_time = {}
    for i in range(len(trigger_seq)):
        ts = first_trg_time + (i * trigger_interval)
        trigger_times.append(ts)
        trg_val = trigger_seq[i][0]
        triggers_by_time[ts] = trg_val

    # Mock the raw_data file
    sep = '\r\n'
    meta = sep.join(['daq_type,TestStream', 'sample_rate,300'])
    header = 'timestamp,c1,c2,c3,TRG'

    data = []
    for i in range(1000):
        timestamp = i + 10.0
        channel_data = [str(random.uniform(-1000, 1000)) for _ in range(3)]
        trg = triggers_by_time.get(timestamp, NONE_VALUE)
        data.append(','.join([str(timestamp), *channel_data, trg]))

    content = sep.join([meta, header, *data])
    return content, trigger_times


class TestTriggers(unittest.TestCase):
    """This is Test Case for Triggers."""

    def test_triggers_for_calibration(self):
        return

    def test_copy_phrase_classifier(self):
        copy_phrase = 'HI'
        typed = 'HI'

        c = CopyPhraseClassifier(copy_phrase, typed)
        self.assertEqual('calib',
                         c.classify("['calibration_trigger', 2.30196808103]"))
        self.assertEqual('fixation', c.classify('+'))
        self.assertEqual('nontarget', c.classify('A'))
        self.assertEqual('nontarget', c.classify('B'))
        self.assertEqual('target', c.classify('H'))
        self.assertEqual('nontarget', c.classify('I'))

        self.assertEqual('fixation', c.classify('+'))
        self.assertEqual('target', c.classify('I'))
        self.assertEqual('nontarget', c.classify('A'))
        self.assertEqual('nontarget', c.classify('H'))
        self.assertEqual('nontarget', c.classify('O'))

    def test_copy_phrase_classifier_correction(self):
        copy_phrase = 'HI'
        typed = 'HA<I'

        c = CopyPhraseClassifier(copy_phrase, typed)
        self.assertEqual('calib',
                         c.classify("['calibration_trigger', 2.30196808103]"))
        self.assertEqual('fixation', c.classify('+'))
        self.assertEqual('nontarget', c.classify('B'))
        self.assertEqual('target', c.classify('H'))
        self.assertEqual('nontarget', c.classify('I'))

        self.assertEqual('fixation', c.classify('+'))
        self.assertEqual('target', c.classify('I'))
        self.assertEqual('nontarget', c.classify('A'))
        self.assertEqual('nontarget', c.classify('O'))

        self.assertEqual('fixation', c.classify('+'))
        self.assertEqual('nontarget', c.classify('I'))
        self.assertEqual('nontarget', c.classify('A'))
        self.assertEqual('target', c.classify('<'))

        self.assertEqual('fixation', c.classify('+'))
        self.assertEqual('target', c.classify('I'))
        self.assertEqual('nontarget', c.classify('A'))
        self.assertEqual('nontarget', c.classify('O'))

    def test_copy_phrase_classifier_correction_double_letters(self):
        copy_phrase = 'HELLO'
        typed = 'HELP<LO'

        c = CopyPhraseClassifier(copy_phrase, typed)
        self.assertEqual('calib',
                         c.classify("['calibration_trigger', 2.30196808103]"))
        # H
        self.assertEqual('fixation', c.classify('+'))
        self.assertEqual('target', c.classify('H'))
        self.assertEqual('nontarget', c.classify('I'))

        # E
        self.assertEqual('fixation', c.classify('+'))
        self.assertEqual('target', c.classify('E'))
        self.assertEqual('nontarget', c.classify('O'))

        # L
        self.assertEqual('fixation', c.classify('+'))
        self.assertEqual('nontarget', c.classify('O'))
        self.assertEqual('target', c.classify('L'))

        # L
        self.assertEqual('fixation', c.classify('+'))
        self.assertEqual('target', c.classify('L'))
        self.assertEqual('nontarget', c.classify('O'))

        # <
        self.assertEqual('fixation', c.classify('+'))
        self.assertEqual('nontarget', c.classify('O'))
        self.assertEqual('target', c.classify('<'))

        # L
        self.assertEqual('fixation', c.classify('+'))
        self.assertEqual('target', c.classify('L'))
        self.assertEqual('nontarget', c.classify('A'))

        # O
        self.assertEqual('fixation', c.classify('+'))
        self.assertEqual('nontarget', c.classify('A'))
        self.assertEqual('target', c.classify('O'))

    def test_extract_from_copy_phrase(self):
        trigger_seq = [
            ("['calibration_trigger', 2.3019680810393766]", 'calib'),
            ('+', 'fixation'), ('I', 'nontarget'), ('H', 'nontarget'),
            ('C', 'nontarget'), ('G', 'nontarget'), ('D', 'nontarget'),
            ('F', 'nontarget'), ('_', 'nontarget'), ('E', 'nontarget'),
            ('<', 'nontarget'), ('B', 'nontarget'), ('+', 'fixation'),
            ('E', 'nontarget'), ('B', 'nontarget'), ('H', 'nontarget'),
            ('_', 'nontarget'), ('C', 'nontarget'), ('I', 'nontarget'),
            ('G', 'nontarget'), ('<', 'nontarget'), ('F', 'nontarget'),
            ('D', 'nontarget'), ('+', 'fixation'), ('G', 'nontarget'),
            ('_', 'nontarget'), ('B', 'nontarget'), ('F', 'nontarget'),
            ('I', 'nontarget'), ('C', 'nontarget'), ('<', 'nontarget'),
            ('E', 'nontarget'), ('D', 'nontarget'), ('H', 'nontarget')]

        phrase = 'HELLO'
        # TODO: why does the copy phrase task starts in the middle of the
        # phrase?
        start_index = int(len(phrase) / 2)
        copy_text = phrase[start_index:]  # 'LLO'
        content, trigger_times = sample_raw_data(trigger_seq)
        extracted = extract_from_copy_phrase(StringIO(content),
                                             copy_text=copy_text,
                                             typed_text='')

        # Assertions
        self.assertEqual(len(trigger_seq), len(extracted))
        for seq_i in range(len(trigger_seq)):
            extracted_val, extracted_targetness, stamp = extracted[seq_i]
            expected_trg, targetness = trigger_seq[seq_i]
            if 'calibration' in expected_trg:
                expected_trg = 'calibration_trigger'
            self.assertEqual(expected_trg, extracted_val)
            self.assertEqual(targetness, extracted_targetness)
            self.assertEqual(trigger_times[seq_i], float(stamp))

    def test_extract_from_calibration(self):
        trigger_seq = [
            ('"[\'calibration_trigger\', 2.039073024992831]"', 'calib'),
            ('J', 'first_pres_target'), ('+', 'fixation'), ('P', 'nontarget'),
            ('R', 'nontarget'), ('E', 'nontarget'), ('K', 'nontarget'),
            ('A', 'nontarget'), ('J', 'target'), ('X', 'nontarget'),
            ('F', 'nontarget'), ('<', 'nontarget'), ('S', 'nontarget'),
            ('E', 'first_pres_target'), ('+', 'fixation'), ('M', 'nontarget'),
            ('T', 'nontarget'), ('H', 'nontarget'), ('W', 'nontarget'),
            ('Y', 'nontarget'), ('V', 'nontarget'), ('E', 'target'),
            ('L', 'nontarget'), ('_', 'nontarget'), ('J', 'nontarget')]

        # Mock the raw_data file
        content, trigger_times = sample_raw_data(trigger_seq)
        extracted = extract_from_calibration(StringIO(content),
                                             seq_len=10)

        # Assertions
        self.assertEqual(len(trigger_seq), len(extracted))
        for seq_i in range(len(trigger_seq)):
            extracted_val, extracted_targetness, stamp = extracted[seq_i]
            expected_trg, targetness = trigger_seq[seq_i]
            if 'calibration' in expected_trg:
                expected_trg = 'calibration_trigger'
            self.assertEqual(expected_trg, extracted_val)
            self.assertEqual(targetness, extracted_targetness)
            self.assertEqual(trigger_times[seq_i], float(stamp))

    def test_writing_trigger_file(self):
        trigger_seq = [
            ('"[\'calibration_trigger\', 2.039073024992831]"', 'calib'),
            ('J', 'first_pres_target'), ('+', 'fixation'), ('P', 'nontarget'),
            ('R', 'nontarget'), ('E', 'nontarget'), ('K', 'nontarget'),
            ('A', 'nontarget'), ('J', 'target'), ('X', 'nontarget'),
            ('F', 'nontarget'), ('<', 'nontarget'), ('S', 'nontarget'),
            ('E', 'first_pres_target'), ('+', 'fixation'), ('M', 'nontarget'),
            ('T', 'nontarget'), ('H', 'nontarget'), ('W', 'nontarget'),
            ('Y', 'nontarget'), ('V', 'nontarget'), ('E', 'target'),
            ('L', 'nontarget'), ('_', 'nontarget'), ('J', 'nontarget')]

        # Mock the raw_data file
        raw_data, trigger_times = sample_raw_data(trigger_seq)
        output = StringIO()
        write_trigger_file_from_lsl_calibration(StringIO(raw_data), output,
                                                seq_len=10)

        written_contents = output.getvalue()
        lines = written_contents.split("\n")

        for i in range(len(lines) - 1):
            written_val, written_targetness, written_stamp = lines[i].split()
            expected_trg, targetness = trigger_seq[i]
            if 'calibration' in expected_trg:
                expected_trg = 'calibration_trigger'
            self.assertEqual(expected_trg, written_val)
            self.assertEqual(targetness, written_targetness)
            self.assertEqual(trigger_times[i], float(written_stamp))


    def test_extract_from_copy_phrase(self):
        # TODO:
        return


if __name__ == '__main__':
    unittest.main()
