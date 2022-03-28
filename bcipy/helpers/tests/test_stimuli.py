import glob
import unittest

from os import path
from mockito import any, mock, unstub, verify, when

from psychopy import core
import numpy as np
import sounddevice as sd
import soundfile as sf

from bcipy.helpers.stimuli import (
    alphabetize,
    best_case_rsvp_inq_gen,
    best_selection,
    calibration_inquiry_generator,
    DEFAULT_FIXATION_PATH,
    get_fixation,
    InquiryReshaper,
    play_sound,
    soundfiles,
    StimuliOrder,
    TrialReshaper,
)

MOCK_FS = 44100


class TestStimuliGeneration(unittest.TestCase):
    """This is Test Case for Stimuli Generated via BciPy."""

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

        # verify all the expected calls happended and the expected number of
        # times
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

        # verify all the expected calls happended and the expected number of
        # times
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

    def test_random_inquiry_gen(self):
        """Test generation of random inquiries"""
        alp = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '<', '_'
        ]
        stim_number = 10
        stim_length = 10
        inquiries, inq_timings, inq_colors = calibration_inquiry_generator(
            alp,
            timing=[0.5, 1, 0.2],
            color=['green', 'red', 'white'],
            stim_number=stim_number,
            stim_length=stim_length,
            stim_order=StimuliOrder.RANDOM,
            is_txt=True)

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

    def test_alphabetical_inquiry_gen(self):
        """Test generation of random inquiries"""
        alp = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '<', '_'
        ]
        stim_number = 10
        stim_length = 10
        inquiries, inq_timings, inq_colors = calibration_inquiry_generator(
            alp,
            timing=[0.5, 1, 0.2],
            color=['green', 'red', 'white'],
            stim_number=stim_number,
            stim_length=stim_length,
            stim_order=StimuliOrder.ALPHABETICAL,
            is_txt=True)

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
        fs = 256
        trial_length_s = 0.5
        reshaped_trials, labels = TrialReshaper()(
            trial_targetness_label=self.target_info,
            timing_info=self.timing_info,
            eeg_data=self.eeg,
            fs=fs,
            channel_map=self.channel_map,
            poststimulus_length=trial_length_s)
        trial_length_samples = int(fs * trial_length_s)
        expected_shape = (self.channel_number, len(self.target_info), trial_length_samples)
        self.assertTrue(np.all(labels == [1, 0, 0]))
        self.assertTrue(reshaped_trials.shape == expected_shape)


class TestInquiryReshaper(unittest.TestCase):
    def setUp(self):
        self.n_channel = 7
        self.trial_length = 0.5
        self.trials_per_inquiry = 3
        self.n_inquiry = 4
        self.fs = 10
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
        self.eeg = np.random.randn(self.n_channel, int(self.fs * (4.8 + self.trial_length)))
        self.channel_map = [1] * self.n_channel

    def test_inquiry_reshaper(self):
        reshaped_data, labels, _ = InquiryReshaper()(
            trial_targetness_label=self.target_info,
            timing_info=self.timing_info,
            eeg_data=self.eeg,
            fs=self.fs,
            trials_per_inquiry=self.trials_per_inquiry,
            channel_map=self.channel_map,
            poststimulus_length=self.trial_length,
        )

        samples_per_inquiry = int(self.fs * self.inquiry_duration_s)
        expected_shape = (self.n_channel, self.n_inquiry, samples_per_inquiry)
        self.assertTrue(reshaped_data.shape == expected_shape)
        self.assertTrue(np.all(labels == self.true_labels))


if __name__ == '__main__':
    unittest.main()
