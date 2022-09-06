import glob
import unittest

from os import path
from mockito import any, mock, unstub, verify, when

from psychopy import core
import numpy as np
import sounddevice as sd
import soundfile as sf

import collections as cnt
from bcipy.helpers.exceptions import BciPyCoreException

from bcipy.helpers.stimuli import (
    alphabetize,
    best_case_rsvp_inq_gen,
    best_selection,
    calibration_inquiry_generator,
    DEFAULT_FIXATION_PATH,
    get_fixation,
    TrialReshaper,
    InquiryReshaper,
    jittered_timing,
    play_sound,
    distributed_target_positions,
    soundfiles,
    StimuliOrder,
    ssvep_to_code,
    TargetPositions
)

MOCK_FS = 44100


class TestStimuliGeneration(unittest.TestCase):
    """This is Test Case for Stimuli Generated via BciPy."""

    alp = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        '<', '_'
    ]

    def tearDown(self):
        unstub()

    def test_calibration_inquiry_generator_with_jitter(self):
        stim_number = 10
        stim_length = 10
        stim_timing = [0.5, 1, 2]
        stim_jitter = 1

        max_jitter = stim_timing[-1] + stim_jitter
        min_jitter = stim_timing[-1] - stim_jitter
        inquiries, inq_timings, inq_colors = calibration_inquiry_generator(
            self.alp,
            timing=stim_timing,
            stim_number=stim_number,
            stim_length=stim_length,
            jitter=stim_jitter)

        self.assertEqual(
            len(inquiries), stim_number,
            'Should have produced the correct number of inquiries')
        self.assertEqual(len(inq_timings), stim_number)
        self.assertEqual(len(inq_colors), stim_number)

        inq_strings = []
        for inq in inquiries:
            self.assertEqual(
                len(inq), stim_length + 2,
                ('inquiry should include the correct number of choices as ',
                 'well as the target and cross.'))
            choices = inq[2:]
            self.assertEqual(stim_length, len(set(choices)),
                             'All choices should be unique')

            # create a string of the options
            inq_strings.append(''.join(choices))

        # ensure timing is jittered
        for j in inq_timings:
            inq_timing = j[2:]  # remove the target presentaion and cross
            for inq_time in inq_timing:
                self.assertTrue(min_jitter <= inq_time <= max_jitter,
                                'Timing should be jittered and within the correct range')

            self.assertTrue(
                len(set(inq_timing)) > 1, 'All choices should be unique')

        self.assertEqual(
            len(inquiries), len(set(inq_strings)),
            'All inquiries should be different')

    def test_calibration_inquiry_generator_random_order(self):
        """Test generation of random inquiries"""
        stim_number = 10
        stim_length = 10
        inquiries, inq_timings, inq_colors = calibration_inquiry_generator(
            self.alp,
            stim_number=stim_number,
            stim_length=stim_length,
            stim_order=StimuliOrder.RANDOM)

        self.assertEqual(
            len(inquiries), stim_number,
            'Should have produced the correct number of inquiries')
        self.assertEqual(len(inq_timings), stim_number)
        self.assertEqual(len(inq_colors), stim_number)

        inq_strings = []
        for inq in inquiries:
            self.assertEqual(
                len(inq), stim_length + 2,
                ('inquiry should include the correct number of choices as ',
                 'well as the target and cross.'))
            choices = inq[2:]
            self.assertEqual(stim_length, len(set(choices)),
                             'All choices should be unique')

            # create a string of the options
            inq_strings.append(''.join(choices))

        self.assertEqual(
            len(inquiries), len(set(inq_strings)),
            'All inquiries should be different')

    def test_calibration_inquiry_generator_alphabetical_order(self):
        """Test generation of random inquiries"""
        stim_number = 10
        stim_length = 10
        inquiries, inq_timings, inq_colors = calibration_inquiry_generator(
            self.alp,
            stim_number=stim_number,
            stim_length=stim_length,
            stim_order=StimuliOrder.ALPHABETICAL)

        self.assertEqual(
            len(inquiries), stim_number,
            'Should have produced the correct number of inquiries')
        self.assertEqual(len(inq_timings), stim_number)
        self.assertEqual(len(inq_colors), stim_number)

        inq_strings = []
        for inq in inquiries:
            self.assertEqual(
                len(inq), stim_length + 2,
                ('inquiry should include the correct number of choices as ',
                 'well as the target and cross.'))
            choices = inq[2:]
            self.assertEqual(stim_length, len(set(choices)),
                             'All choices should be unique')
            self.assertEqual(alphabetize(choices), choices)

            # create a string of the options
            inq_strings.append(''.join(choices))

        self.assertEqual(
            len(inquiries), len(set(inq_strings)),
            'All inquiries should be different')

    def test_calibration_inquiry_generator_distributed_targets(self):
        """Test generation of inquiries with distributed target positions"""
        stim_number = 10
        stim_length = 10
        nontarget_inquiries = 10
        inquiries, inq_timings, inq_colors = calibration_inquiry_generator(
            self.alp,
            stim_number=stim_number,
            stim_length=stim_length,
            stim_order=StimuliOrder.RANDOM,
            target_positions=TargetPositions.DISTRIBUTED,
            nontarget_inquiries=nontarget_inquiries)

        self.assertEqual(
            len(inquiries), stim_number,
            'Should have produced the correct number of inquiries')
        self.assertEqual(len(inq_timings), stim_number)
        self.assertEqual(len(inq_colors), stim_number)

        inq_strings = []
        for inq in inquiries:
            self.assertEqual(
                len(inq), stim_length + 2,
                ('inquiry should include the correct number of choices as ',
                 'well as the target and cross.'))
            choices = inq[2:]
            self.assertEqual(stim_length, len(set(choices)),
                             'All choices should be unique')

            # create a string of the options
            inq_strings.append(''.join(choices))

        self.assertEqual(
            len(inquiries), len(set(inq_strings)),
            'All inquiries should be different')

    def test_calibration_inquiry_generator_distributed_targets_alphabetical(self):
        """Test generation of inquiries with distributed target positions"""
        stim_number = 10
        stim_length = 10
        nontarget_inquiries = 20
        inquiries, inq_timings, inq_colors = calibration_inquiry_generator(
            self.alp,
            stim_number=stim_number,
            stim_length=stim_length,
            stim_order=StimuliOrder.ALPHABETICAL,
            target_positions=TargetPositions.DISTRIBUTED,
            nontarget_inquiries=nontarget_inquiries)

        self.assertEqual(
            len(inquiries), stim_number,
            'Should have produced the correct number of inquiries')
        self.assertEqual(len(inq_timings), stim_number)
        self.assertEqual(len(inq_colors), stim_number)

        inq_strings = []
        for inq in inquiries:
            self.assertEqual(
                len(inq), stim_length + 2,
                ('inquiry should include the correct number of choices as ',
                 'well as the target and cross.'))
            choices = inq[2:]
            self.assertEqual(stim_length, len(set(choices)),
                             'All choices should be unique')

            # create a string of the options
            inq_strings.append(''.join(choices))

        self.assertEqual(
            len(inquiries), len(set(inq_strings)),
            'All inquiries should be different')

    def test_calibration_inquiry_generator_distributed_targets_no_nontargets(self):
        """Test generation of inquiries with distributed target positions and no nontarget inquiries."""
        stim_number = 10
        stim_length = 10
        nontarget_inquiries = 0
        inquiries, inq_timings, inq_colors = calibration_inquiry_generator(
            self.alp,
            stim_number=stim_number,
            stim_length=stim_length,
            stim_order=StimuliOrder.RANDOM,
            target_positions=TargetPositions.DISTRIBUTED,
            nontarget_inquiries=nontarget_inquiries)

        self.assertEqual(
            len(inquiries), stim_number,
            'Should have produced the correct number of inquiries')
        self.assertEqual(len(inq_timings), stim_number)
        self.assertEqual(len(inq_colors), stim_number)

        inq_strings = []
        for inq in inquiries:
            self.assertEqual(
                len(inq), stim_length + 2,
                ('inquiry should include the correct number of choices as ',
                 'well as the target and cross.'))
            choices = inq[2:]
            self.assertEqual(stim_length, len(set(choices)),
                             'All choices should be unique')

            # create a string of the options
            inq_strings.append(''.join(choices))

        self.assertEqual(
            len(inquiries), len(set(inq_strings)),
            'All inquiries should be different')

    def test_calibration_inquiry_generator_distributed_targets_positions(self):
        """Test generation of distributed target positions with nontarget inquiries."""

        stim_number = 11
        stim_length = 10
        nontarget_inquiries = 10

        nontarget_inquiry = (int)(stim_number * (nontarget_inquiries / 100))
        target_inquiries = stim_number - nontarget_inquiry
        num_target_inquiries = (int)(target_inquiries / stim_length)

        targets = distributed_target_positions(stim_number=stim_number,
                                               stim_length=stim_length,
                                               nontarget_inquiries=nontarget_inquiries)

        self.assertEqual(
            len(targets), stim_number,
            'Should have produced the correct number of targets for inquiries.')

        # count how many times each target position is used
        count = cnt.Counter()
        for pos in targets:
            count[pos] += 1

        # make sure position counts are equally distributed, including non-target
        for i in count:
            self.assertTrue(num_target_inquiries <= count[i] <= num_target_inquiries + 1)

    def test_calibration_inquiry_generator_distributed_targets_positions_half_nontarget(self):
        """Test generation of distributed target positions with half being nontarget inquiries."""

        stim_number = 120
        stim_length = 9
        nontarget_inquiries = 50

        nontarget_inquiry = (int)(stim_number * (nontarget_inquiries / 100))
        target_inquiries = stim_number - nontarget_inquiry
        num_target_inquiries = (int)(target_inquiries / stim_length)

        targets = distributed_target_positions(stim_number=stim_number,
                                               stim_length=stim_length,
                                               nontarget_inquiries=nontarget_inquiries)

        self.assertEqual(
            len(targets), stim_number,
            'Should have produced the correct number of targets for inquiries.')

        # count how many times each target position is used
        count = cnt.Counter()
        for pos in targets:
            count[pos] += 1

        # make sure target position counts are equally distributed
        for i in count:
            if i is not None:
                self.assertTrue(num_target_inquiries <= count[i] <= num_target_inquiries + 1)

        # make sure correct number of non-target inquiries
        self.assertEqual(count[None], nontarget_inquiry,
                         'Should have produced 50 percent of 120 non-target positions.')

    def test_calibration_inquiry_generator_distributed_targets_positions_no_nontargets(self):
        """Test generation of distributed target positions with no nontarget inquiries."""

        stim_number = 50
        stim_length = 11
        nontarget_inquiries = 0

        nontarget_inquiry = (int)(stim_number * (nontarget_inquiries / 100))
        target_inquiries = stim_number - nontarget_inquiry
        num_target_inquiries = (int)(target_inquiries / stim_length)

        targets = distributed_target_positions(stim_number=stim_number,
                                               stim_length=stim_length,
                                               nontarget_inquiries=nontarget_inquiries)

        self.assertEqual(
            len(targets), stim_number,
            'Should have produced the correct number of targets for inquiries.')

        # count how many times each target position is used
        count = cnt.Counter()
        for pos in targets:
            count[pos] += 1

        # make sure target position counts are equally distributed
        for i in count:
            if i is not None:
                self.assertTrue(num_target_inquiries <= count[i] <= num_target_inquiries + 1)

        # make sure there are no non-target inquiries
        self.assertEqual(count[None], 0,
                         'Should have produced no non-target positions.')

    def test_calibration_inquiry_generator_distributed_targets_all_nontargets(self):
        """Test generation of distributed target positions with all inquiries being non-target."""

        stim_number = 100
        stim_length = 6
        nontarget_inquiries = 100

        nontarget_inquiry = (int)(stim_number * (nontarget_inquiries / 100))
        target_inquiries = stim_number - nontarget_inquiry
        num_target_inquiries = (int)(target_inquiries / stim_length)

        targets = distributed_target_positions(
            stim_number=stim_number,
            stim_length=stim_length,
            nontarget_inquiries=nontarget_inquiries)

        self.assertEqual(
            len(targets), stim_number,
            'Should have produced the correct number of targets for inquiries.')

        # count how many times each target position is used
        count = cnt.Counter()
        for pos in targets:
            count[pos] += 1

        # make sure target position counts are equally distributed
        for i in count:
            if i is not None:
                self.assertTrue(num_target_inquiries <= count[i] <= num_target_inquiries + 1)

        # make sure all inquries are non-target inquiries
        self.assertEqual(count[None], stim_number,
                         'Should have produced all non-target positions.')

    def test_best_selection(self):
        """Test best_selection"""

        self.assertEqual(['a', 'c', 'e'],
                         best_selection(
                             selection_elements=['a', 'b', 'c', 'd', 'e'],
                             val=[0.3, 0.1, 0.3, 0.1, 0.2],
                             len_query=3))

        # Test equal probabilities
        self.assertEqual(['a', 'b', 'c'],
                         best_selection(
                             selection_elements=['a', 'b', 'c', 'd', 'e'],
                             val=[0.2, 0.2, 0.2, 0.2, 0.2],
                             len_query=3))

        # Test always included
        self.assertEqual(
            ['a', 'c', 'd'],
            best_selection(
                selection_elements=['a', 'b', 'c', 'd', 'e'],
                val=[0.3, 0.1, 0.3, 0.1, 0.2],
                len_query=3,
                always_included=['d']),
            'Included item should bump out the best item with the lowest val.')

        self.assertEqual(
            ['a', 'b', 'c'],
            best_selection(
                selection_elements=['a', 'b', 'c', 'd', 'e'],
                val=[0.5, 0.4, 0.1, 0.0, 0.0],
                len_query=3,
                always_included=['b']),
            'Included item should retain its position if already present')

        self.assertEqual(['a', 'b', 'e'],
                         best_selection(
                             selection_elements=['a', 'b', 'c', 'd', 'e'],
                             val=[0.5, 0.0, 0.1, 0.3, 0.0],
                             len_query=3,
                             always_included=['b', 'e']),
                         'multiple included items should be supported.')

        self.assertEqual(['d'],
                         best_selection(
                             selection_elements=['a', 'b', 'c', 'd', 'e'],
                             val=[0.5, 0.4, 0.1, 0.0, 0.0],
                             len_query=1,
                             always_included=['d', 'e']),
                         'len_query should be respected.')

        self.assertEqual(['a', 'b', 'c'],
                         best_selection(
            selection_elements=['a', 'b', 'c', 'd', 'e'],
            val=[0.5, 0.4, 0.1, 0.0, 0.0],
            len_query=3,
            always_included=['<']),
            'should ignore items not in the set.')

    def test_best_case_inquiry_gen(self):
        """Test best_case_rsvp_inq_gen"""
        alp = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        n = 5
        samples, times, colors = best_case_rsvp_inq_gen(
            alp=alp,
            session_stimuli=[0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2],
            timing=[1, 0.2],
            color=['red', 'white'],
            stim_number=1,
            stim_length=n,
            is_txt=True)

        first_inq = samples[0]
        self.assertEqual(1, len(samples))
        self.assertEqual(n + 1, len(first_inq),
                         'Should include fixation cross.')
        self.assertEqual(len(samples), len(times))
        self.assertEqual(len(samples), len(colors))

        expected = ['+', 'a', 'b', 'd', 'e', 'g']
        for letter in expected:
            self.assertTrue(letter in first_inq)

        self.assertNotEqual(expected, first_inq, 'Should be in random order.')
        self.assertEqual([1] + ([0.2] * n), times[0])
        self.assertEqual(['red'] + (['white'] * n), colors[0])

    def test_best_case_inquiry_gen_with_inq_constants(self):
        """Test best_case_rsvp_inq_gen with inquiry constants"""

        alp = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        n = 5

        with self.assertRaises(
                Exception, msg='Constants should be in the alphabet'):
            best_case_rsvp_inq_gen(
                alp=alp,
                session_stimuli=[0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2],
                inq_constants=['<'])

        alp = ['a', 'b', 'c', 'd', 'e', 'f', 'g', '<']
        samples, times, colors = best_case_rsvp_inq_gen(
            alp=alp,
            session_stimuli=[0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2, 0.0],
            stim_number=1,
            stim_length=n,
            is_txt=True,
            inq_constants=['<'])

        first_inq = samples[0]
        self.assertEqual(1, len(samples))
        self.assertEqual(n + 1, len(first_inq),
                         'Should include fixation cross.')
        self.assertEqual(len(samples), len(times))
        self.assertEqual(len(samples), len(colors))

        expected = ['+', 'a', 'd', 'e', 'g', '<']
        for letter in expected:
            self.assertTrue(letter in first_inq)

        self.assertNotEqual(expected, first_inq, 'Should be in random order.')
        self.assertEqual([1] + ([0.2] * n), times[0])
        self.assertEqual(['red'] + (['white'] * n), colors[0])


class TestJitteredTiming(unittest.TestCase):

    def test_jittered_timing_returns_correct_number_of_stims(self):
        time = 1
        jitter = 0.24
        stim_number = 10

        resp = jittered_timing(time, jitter, stim_number)

        self.assertEqual(stim_number, len(resp))
        self.assertIsInstance(resp, list)

    def test_jittered_timing_with_jitter_withint_defined_limits(self):
        time = 1
        jitter = 0.25
        stim_number = 100

        max_jitter = time + jitter
        min_jitter = time - jitter

        resp = jittered_timing(time, jitter, stim_number)

        for r in resp:
            self.assertTrue(min_jitter <= r <= max_jitter)

    def test_jittered_timing_throw_exception_when_jitter_greater_than_time(self):
        # to prevent 0 values we prevent the jitter from being greater than the time
        time = 1
        jitter = 1.5
        stim_number = 100

        with self.assertRaises(Exception, msg='Jitter should be less than stimuli time'):
            jittered_timing(time, jitter, stim_number)


class TestGetFixation(unittest.TestCase):

    def test_text_fixation(self):
        expected = '+'
        response = get_fixation(is_txt=True)
        self.assertEqual(expected, response)

    def test_image_fixation_uses_default(self):
        expected = DEFAULT_FIXATION_PATH
        response = get_fixation(is_txt=False)
        self.assertEqual(expected, response)


class TestAlphabetize(unittest.TestCase):

    def setUp(self) -> None:
        self.list_to_alphabetize = ['Z', 'Q', 'A', 'G']

    def test_alphabetize(self):
        expected = ['A', 'G', 'Q', 'Z']
        response = alphabetize(self.list_to_alphabetize)
        self.assertEqual(expected, response)

    def test_alphabetize_image_name(self):
        list_of_images = ['testing.png', '_ddtt.jpeg', 'bci_image.png']
        expected = ['bci_image.png', 'testing.png', '_ddtt.jpeg']
        response = alphabetize(list_of_images)
        self.assertEqual(expected, response)

    def test_alphabetize_special_characters_at_end(self):
        character = '<'
        expected = ['A', 'G', 'Q', 'Z', character]
        self.list_to_alphabetize.insert(1, character)
        response = alphabetize(self.list_to_alphabetize)
        self.assertEqual(expected, response)


class TestTrialReshaper(unittest.TestCase):
    def setUp(self):
        self.target_info = ['target', 'nontarget', 'nontarget']
        self.timing_info = [1.001, 1.2001, 1.4001]
        # make some fake eeg data
        self.channel_number = 21
        tmp_inp = np.array([range(4000)] * self.channel_number)
        # Add some channel info
        tmp_inp[:, 0] = np.transpose(np.arange(1, self.channel_number + 1, 1))
        self.eeg = tmp_inp
        self.channel_map = [1] * self.channel_number

    def test_trial_reshaper(self):
        sample_rate = 256
        trial_length_s = 0.5
        reshaped_trials, labels = TrialReshaper()(
            trial_targetness_label=self.target_info,
            timing_info=self.timing_info,
            eeg_data=self.eeg,
            sample_rate=sample_rate,
            channel_map=self.channel_map,
            poststimulus_length=trial_length_s)
        trial_length_samples = int(sample_rate * trial_length_s)
        expected_shape = (self.channel_number, len(self.target_info), trial_length_samples)
        self.assertTrue(np.all(labels == [1, 0, 0]))
        self.assertTrue(reshaped_trials.shape == expected_shape)


class TestInquiryReshaper(unittest.TestCase):
    def setUp(self):
        self.n_channel = 7
        self.trial_length = 0.5
        self.trials_per_inquiry = 3
        self.n_inquiry = 4
        self.sample_rate = 10
        self.target_info = [
            "target", "nontarget", "nontarget",
            "nontarget", "nontarget", "nontarget",
            "nontarget", "target", "nontarget",
            "nontarget", "nontarget", "target",
        ]
        self.true_labels = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        self.timing_info = [
            1.4, 1.6, 1.8,
            2.4, 2.6, 2.8,
            3.4, 3.6, 3.8,
            4.4, 4.6, 4.8,
        ]
        # Inquiry lasts from onset of first trial, to onset of last trial + trial_length
        self.inquiry_duration_s = 1.8 - 1.4 + self.trial_length

        # total duration = 3.8s + final trial_length
        self.eeg = np.random.randn(self.n_channel, int(self.sample_rate * (4.8 + self.trial_length)))
        self.channel_map = [1] * self.n_channel

    def test_inquiry_reshaper(self):
        reshaped_data, labels, _ = InquiryReshaper()(
            trial_targetness_label=self.target_info,
            timing_info=self.timing_info,
            eeg_data=self.eeg,
            sample_rate=self.sample_rate,
            trials_per_inquiry=self.trials_per_inquiry,
            channel_map=self.channel_map,
            poststimulus_length=self.trial_length,
        )

        samples_per_inquiry = int(self.sample_rate * self.inquiry_duration_s)
        expected_shape = (self.n_channel, self.n_inquiry, samples_per_inquiry)
        self.assertTrue(reshaped_data.shape == expected_shape)
        self.assertTrue(np.all(labels == self.true_labels))


class SSVEPStimuli(unittest.TestCase):

    def test_default_flicker_and_refresh_rate_return_codes(self):
        response = ssvep_to_code()
        self.assertIsInstance(response, list)
        self.assertEqual(response[0], 0)

    def test_ssvep_to_codes_returns_the_length_of_refresh_rate(self):
        refresh_rate = 40
        flicker_rate = 2
        response = ssvep_to_code(flicker_rate=flicker_rate, refresh_rate=refresh_rate)
        self.assertTrue(len(response) == refresh_rate)
        self.assertEqual(response[0], 0)
        self.assertEqual(response[-1], 1)

    def test_ssvep_to_code_raises_exception_when_refresh_rate_less_than_flicker_rate(self):
        flicker_rate = 300
        refresh_rate = 1

        with self.assertRaises(BciPyCoreException):
            ssvep_to_code(refresh_rate, flicker_rate)

    def test_when_division_of_refresh_rate_by_flicker_rate_raise_exception_if_noninteger(self):
        flicker_rate = 11
        refresh_rate = 60

        with self.assertRaises(BciPyCoreException):
            ssvep_to_code(refresh_rate, flicker_rate)

    def test_ssvep_to_code_returns_expected_codes(self):
        flicker_rate = 2
        refresh_rate = 4
        response = ssvep_to_code(flicker_rate=flicker_rate, refresh_rate=refresh_rate)
        expected_output = [0, 0, 1, 1]
        self.assertEqual(response, expected_output)

    def test_ssvep_to_code_raises_exception_when_flicker_rate_one_or_less(self):
        flicker_rate = 1
        refresh_rate = 2
        with self.assertRaises(BciPyCoreException):
            ssvep_to_code(refresh_rate, flicker_rate)

        flicker_rate = 0
        refresh_rate = 2
        with self.assertRaises(BciPyCoreException):
            ssvep_to_code(refresh_rate, flicker_rate)


class TestSoundStimuli(unittest.TestCase):

    def tearDown(self):
        unstub()

    def test_play_sound_returns_timing(self):
        # fake sound file path
        sound_file_path = 'test_sound_file_path'

        # mock the other library interactions
        when(sf).read(
            sound_file_path, dtype='float32').thenReturn(('data', MOCK_FS))
        when(sd).play(any(), any()).thenReturn(None)
        when(core).wait(any()).thenReturn(None)

        # play our test sound file
        timing = play_sound(sound_file_path)

        # assert the response is as expected
        self.assertIsInstance(timing, list)

        # verify all the expected calls happended and the expected number of times
        verify(sf, times=1).read(sound_file_path, dtype='float32')
        verify(sd, times=1).play('data', MOCK_FS)
        verify(core, times=2).wait(any())

    def test_play_sound_raises_exception_if_soundfile_cannot_read_file(self):
        # fake sound file path
        sound_file_path = 'test_sound_file_path'

        # mock the other library interactions
        when(sf).read(
            sound_file_path, dtype='float32').thenRaise(Exception(''))

        # assert it raises the exception
        with self.assertRaises(Exception):
            play_sound(sound_file_path)

        # verify all the expected calls happended and the expected number of times
        verify(sf, times=1).read(sound_file_path, dtype='float32')

    def test_play_sound_sound_callback_evokes_with_timing(self):
        # fake sound file path
        sound_file_path = 'test_sound_file_path'
        test_trigger_name = 'test_trigger_name'
        test_trigger_time = 111
        self.test_timing = [test_trigger_name, test_trigger_time]

        experiment_clock = mock()

        def mock_callback_function(clock, stimuli):
            self.assertEqual(stimuli, self.test_timing[0])

        # mock the other library interactions
        when(sf).read(
            sound_file_path, dtype='float32').thenReturn(('data', MOCK_FS))
        when(sd).play(any(), any()).thenReturn(None)
        when(core).wait(any()).thenReturn(None)
        when(experiment_clock).getTime().thenReturn(test_trigger_time)

        play_sound(
            sound_file_path,
            track_timing=True,
            sound_callback=mock_callback_function,
            trigger_name=test_trigger_name,
            experiment_clock=experiment_clock,
        )

        # verify all the expected calls happended and the expected number of times
        verify(sf, times=1).read(sound_file_path, dtype='float32')
        verify(sd, times=1).play('data', MOCK_FS)
        verify(core, times=2).wait(any())

    def test_soundfiles_generator(self):
        """Test that soundfiles function returns an cyclic generator."""

        directory = path.join('.', 'sounds')
        soundfile_paths = [
            path.join(directory, '0.wav'),
            path.join(directory, '1.wav'),
            path.join(directory, '2.wav')
        ]
        when(glob).glob(
            path.join(
                directory,
                '*.wav')).thenReturn(soundfile_paths)
        when(path).isdir(directory).thenReturn(True)

        gen = soundfiles(directory)
        self.assertEqual(next(gen), soundfile_paths[0])
        self.assertEqual(next(gen), soundfile_paths[1])
        self.assertEqual(next(gen), soundfile_paths[2])
        self.assertEqual(next(gen), soundfile_paths[0])
        self.assertEqual(next(gen), soundfile_paths[1])
        for _ in range(10):
            self.assertTrue(next(gen) in soundfile_paths)

    def test_soundfiles_generator_path_arg(self):
        """Test that soundfiles function constructs the correct path."""
        directory = path.join('.', 'sounds')
        soundfile_paths = [
            path.join(directory, '0.wav'),
            path.join(directory, '1.wav'),
            path.join(directory, '2.wav')
        ]
        when(glob).glob(
            path.join(
                directory,
                '*.wav')).thenReturn(soundfile_paths)
        when(path).isdir(directory).thenReturn(True)
        gen = soundfiles(directory)
        self.assertEqual(next(gen), soundfile_paths[0])


if __name__ == '__main__':
    unittest.main()
