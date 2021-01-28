import unittest
from io import StringIO
from typing import List, Tuple

from bcipy.helpers.parameters import Parameters
from bcipy.helpers.triggers import (NONE_VALUES, LslCopyPhraseLabeller,
                                    extract_from_calibration,
                                    extract_from_copy_phrase, read_triggers,
                                    trigger_durations,
                                    write_trigger_file_from_lsl_calibration)
from bcipy.signal.generator.generator import gen_random_data


def sample_raw_data(trigger_seq: List[Tuple[str, str]] = [],
                    first_trg_time: int = 100,
                    trigger_interval: int = 10,
                    daq_type: str = 'TestStream',
                    sample_rate: int = 300,
                    ch_names: List[str] = ['c1', 'c2',
                                           'c3']) -> Tuple[str, List[float]]:
    """Helper function for creating mock data that looks like the raw_data.csv
    output. Adds trigger data to the TRG column at the specified interval.

    Parameters:
    -----------
        trigger_seq: list of tuples with the stimulus, targetness.
        first_trg_time: first time in the data where a trigger should appear.
        trigger_interval: set interval at which subsequent triggers should be
                    displayed
        daq_type - metadata written to the raw_data file.
        sample_rate - metadata written to the raw_data file for sample rate
            in hz.
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
    meta = sep.join([f'daq_type,{daq_type}', 'sample_rate,{sample_rate}'])
    header = 'timestamp,' + ','.join(ch_names) + ',TRG'

    data = []
    n_channels = len(ch_names)
    for i in range(1000):
        timestamp = i + 10.0
        channel_data = list(map(str, gen_random_data(-1000, 1000, n_channels)))
        trg = triggers_by_time.get(timestamp, NONE_VALUES[0])
        data.append(','.join([str(timestamp), *channel_data, trg]))

    content = sep.join([meta, header, *data])
    return content, trigger_times


class TestTriggers(unittest.TestCase):
    """This is Test Case for Triggers."""

    def test_triggers_for_calibration(self):
        return

    def test_copy_phrase_labeller(self):
        copy_phrase = 'HI'
        typed = 'HI'

        labeller = LslCopyPhraseLabeller(copy_phrase, typed)
        self.assertEqual(
            'calib', labeller.label("['calibration_trigger', "
                                    "2.30196808103]"))
        self.assertEqual('fixation', labeller.label('+'))
        self.assertEqual('nontarget', labeller.label('A'))
        self.assertEqual('nontarget', labeller.label('B'))
        self.assertEqual('target', labeller.label('H'))
        self.assertEqual('nontarget', labeller.label('I'))

        self.assertEqual('fixation', labeller.label('+'))
        self.assertEqual('target', labeller.label('I'))
        self.assertEqual('nontarget', labeller.label('A'))
        self.assertEqual('nontarget', labeller.label('H'))
        self.assertEqual('nontarget', labeller.label('O'))

    def test_copy_phrase_labeller_correction(self):
        copy_phrase = 'HI'
        typed = 'HA<I'

        labeller = LslCopyPhraseLabeller(copy_phrase, typed)
        self.assertEqual(
            'calib', labeller.label("['calibration_trigger', "
                                    "2.30196808103]"))
        self.assertEqual('fixation', labeller.label('+'))
        self.assertEqual('nontarget', labeller.label('B'))
        self.assertEqual('target', labeller.label('H'))
        self.assertEqual('nontarget', labeller.label('I'))

        self.assertEqual('fixation', labeller.label('+'))
        self.assertEqual('target', labeller.label('I'))
        self.assertEqual('nontarget', labeller.label('A'))
        self.assertEqual('nontarget', labeller.label('O'))

        self.assertEqual('fixation', labeller.label('+'))
        self.assertEqual('nontarget', labeller.label('I'))
        self.assertEqual('nontarget', labeller.label('A'))
        self.assertEqual('target', labeller.label('<'))

        self.assertEqual('fixation', labeller.label('+'))
        self.assertEqual('target', labeller.label('I'))
        self.assertEqual('nontarget', labeller.label('A'))
        self.assertEqual('nontarget', labeller.label('O'))

    def test_copy_phrase_labeller_correction_double_letters(self):
        copy_phrase = 'HELLO'
        typed = 'HELP<LO'

        labeller = LslCopyPhraseLabeller(copy_phrase, typed)
        self.assertEqual(
            'calib', labeller.label("['calibration_trigger', "
                                    "2.30196808103]"))
        # H
        self.assertEqual('fixation', labeller.label('+'))
        self.assertEqual('target', labeller.label('H'))
        self.assertEqual('nontarget', labeller.label('I'))

        # E
        self.assertEqual('fixation', labeller.label('+'))
        self.assertEqual('target', labeller.label('E'))
        self.assertEqual('nontarget', labeller.label('O'))

        # L
        self.assertEqual('fixation', labeller.label('+'))
        self.assertEqual('nontarget', labeller.label('O'))
        self.assertEqual('target', labeller.label('L'))

        # L
        self.assertEqual('fixation', labeller.label('+'))
        self.assertEqual('target', labeller.label('L'))
        self.assertEqual('nontarget', labeller.label('O'))

        # <
        self.assertEqual('fixation', labeller.label('+'))
        self.assertEqual('nontarget', labeller.label('O'))
        self.assertEqual('target', labeller.label('<'))

        # L
        self.assertEqual('fixation', labeller.label('+'))
        self.assertEqual('target', labeller.label('L'))
        self.assertEqual('nontarget', labeller.label('A'))

        # O
        self.assertEqual('fixation', labeller.label('+'))
        self.assertEqual('nontarget', labeller.label('A'))
        self.assertEqual('target', labeller.label('O'))

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
            ('E', 'nontarget'), ('D', 'nontarget'), ('H', 'nontarget')
        ]

        phrase = 'HELLO'
        # TODO: why does the copy phrase task starts in the middle of the
        # phrase?
        start_index = int(len(phrase) / 2)
        copy_text = phrase[start_index:]  # 'LLO'
        content, trigger_times = sample_raw_data(trigger_seq)
        extracted = extract_from_copy_phrase(StringIO(content),
                                             copy_text=copy_text,
                                             typed_text=copy_text)

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
            ('L', 'nontarget'), ('_', 'nontarget'), ('J', 'nontarget')
        ]

        # Mock the raw_data file
        content, trigger_times = sample_raw_data(trigger_seq)
        extracted = extract_from_calibration(StringIO(content), inq_len=10)

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
            ('L', 'nontarget'), ('_', 'nontarget'), ('J', 'nontarget')
        ]

        # Mock the raw_data file
        raw_data, trigger_times = sample_raw_data(trigger_seq)
        output = StringIO()
        write_trigger_file_from_lsl_calibration(StringIO(raw_data),
                                                output,
                                                inq_len=10)

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

    def test_trigger_durations(self):
        """Test trigger durations"""

        parameters = Parameters.from_cast_values(time_target=1.0,
                                                 time_cross=0.5,
                                                 time_flash=0.2)
        durations = trigger_durations(parameters)

        self.assertEqual(durations['calib'], 0.0)
        self.assertEqual(durations['first_pres_target'], 1.0)
        self.assertEqual(durations['fixation'], 0.5)
        self.assertEqual(durations['nontarget'], 0.2)
        self.assertEqual(durations['target'], 0.2)

    def test_read_triggers(self):
        """Test reading in triggers from a file."""
        trg_data = '''calibration_trigger calib 3.4748408449813724
J first_pres_target 6.151848723005969
+ fixation 8.118640798988054
F nontarget 8.586895030981395
D nontarget 8.887798132986063
J target 9.18974666899885
T nontarget 9.496583286992973
K nontarget 9.798354075988755
Q nontarget 10.099591801001225
O nontarget 10.401458177977474
Z nontarget 10.70310750597855
R nontarget 11.00485198898241
_ nontarget 11.306160968990298
W first_pres_target 13.155240687978221
+ fixation 15.122089709999273
N nontarget 15.58976313797757
B nontarget 15.891450178984087
W target 16.192583801981527
P nontarget 16.49438149499474
C nontarget 16.795942058990477
Y nontarget 17.09710298400023
Q nontarget 17.398642276995815
A nontarget 17.699613840988604
F nontarget 18.000999594980385
J nontarget 18.302860347001115
offset offset_correction 6.23828125
'''

        data = read_triggers(StringIO(trg_data))
        self.assertEqual(len(data), 25)
        calib = data[0]
        self.assertEqual(calib[0], 'calibration_trigger')
        self.assertEqual(calib[1], 'calib')
        self.assertEqual(type(calib[2]), float)
        self.assertTrue(calib[2] > 3.47 and calib[2] < 7,
                        "Should account for offset")


if __name__ == '__main__':
    unittest.main()
