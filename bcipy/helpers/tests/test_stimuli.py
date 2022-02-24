import glob
import unittest
from os import path
import sounddevice as sd
import soundfile as sf
from mockito import any, mock, unstub, verify, when
from psychopy import core

import numpy as np

from bcipy.helpers.stimuli import (
    TargetPositions,
    alphabetize,
    best_case_rsvp_inq_gen,
    best_selection,
    DEFAULT_FIXATION_PATH,
    get_fixation,
    play_sound,
    calibration_inquiry_generator,
    distributed_target_positions,
    soundfiles,
    StimuliOrder,
    TargetPositions
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

        # verify all the expected calls happended and the expected number of
        # times
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
            target_positions=TargetPositions.RANDOM,
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
            target_positions=TargetPositions.RANDOM,
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

    def test_distributed_target_inquiry_gen(self):
        """Test generation of inquiries with distributed target positions"""
        alp = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '<', '_'
        ]
        stim_number = 100
        stim_length = 10
        nontarget_inquiries = 10
        inquiries, inq_timings, inq_colors = calibration_inquiry_generator(
            alp,
            timing=[0.5, 1, 0.2],
            color=['green', 'red', 'white'],
            stim_number=stim_number,
            stim_length=stim_length,
            stim_order=StimuliOrder.RANDOM,
            target_positions=TargetPositions.DISTRIBUTED,
            nontarget_inquiries=nontarget_inquiries,
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

    def test_distributed_target_inquiry_gen_no_nontarget(self):
        """Test generation of inquiries with distributed target positions"""
        alp = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '<', '_'
        ]
        stim_number = 100
        stim_length = 10
        nontarget_inquiries = 0
        inquiries, inq_timings, inq_colors = calibration_inquiry_generator(
            alp,
            timing=[0.5, 1, 0.2],
            color=['green', 'red', 'white'],
            stim_number=stim_number,
            stim_length=stim_length,
            stim_order=StimuliOrder.RANDOM,
            target_positions=TargetPositions.DISTRIBUTED,
            nontarget_inquiries=nontarget_inquiries,
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
    
    def test_distributed_target_inquiry_gen_all_nontarget(self):
        """Test generation of inquiries with distributed target positions"""
        alp = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '<', '_'
        ]
        stim_number = 100
        stim_length = 10
        nontarget_inquiries = 100
        inquiries, inq_timings, inq_colors = calibration_inquiry_generator(
            alp,
            timing=[0.5, 1, 0.2],
            color=['green', 'red', 'white'],
            stim_number=stim_number,
            stim_length=stim_length,
            stim_order=StimuliOrder.RANDOM,
            target_positions=TargetPositions.DISTRIBUTED,
            nontarget_inquiries=nontarget_inquiries,
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

    def test_distributed_target_positions(self):
        """Test generation of distributed target positions"""

        stim_number = 11
        stim_length = 10
        nontarget_inquiries = 10

        target_inquiries = stim_number - (stim_number * (nontarget_inquiries / 100))
        num_target_inquiries = (int) (target_inquiries / stim_length)

        targets = distributed_target_positions(stim_number=stim_number, 
            stim_length=stim_length, 
            nontarget_inquiries=nontarget_inquiries)

        self.assertEqual(
            len(targets), stim_number,
            'Should have produced the correct number of targets for inquiries.')

        target_counts = np.zeros(stim_length + 1)
        target_counts = target_counts.astype(int)

        #count how many times each target position is used
        for pos in targets:
            target_counts[pos] = (target_counts[pos] + 1)

        #make sure position counts are equally distributed, including non-target
        for i in target_counts:
            self.assertTrue(i >= num_target_inquiries)
            self.assertTrue(i <= (num_target_inquiries + 1))

    def test_distributed_target_positions_no_nontarget_inquiries(self):
        """Test generation of distributed target positions"""

        stim_number = 50
        stim_length = 10
        nontarget_inquiries = 0

        target_inquiries = stim_number - (stim_number * (nontarget_inquiries / 100))
        num_target_inquiries = (int) (target_inquiries / stim_length)

        targets = distributed_target_positions(stim_number=stim_number, 
            stim_length=stim_length, 
            nontarget_inquiries=nontarget_inquiries)

        self.assertEqual(
            len(targets), stim_number,
            'Should have produced the correct number of targets for inquiries.')

        target_counts = np.zeros(stim_length + 1)
        target_counts = target_counts.astype(int)

        #count how many times each target position is used
        for pos in targets:
            target_counts[pos] = (target_counts[pos] + 1)

        #make sure position counts are equally distributed
        for i in (target_counts[0:stim_length]):
            self.assertTrue(i >= num_target_inquiries)
            self.assertTrue(i <= (num_target_inquiries + 1))

        self.assertEqual(target_counts[stim_length], 0, 
            'Should have produced no non-target positions.')

    def test_distributed_target_positions_all_nontarget_inquiries(self):
        """Test generation of distributed target positions"""

        stim_number = 100
        stim_length = 9
        nontarget_inquiries = 100

        target_inquiries = stim_number - (stim_number * (nontarget_inquiries / 100))
        print(target_inquiries)
        num_target_inquiries = (int) (target_inquiries / stim_length)
        print(num_target_inquiries)

        targets = distributed_target_positions(stim_number=stim_number, stim_length=stim_length, nontarget_inquiries=nontarget_inquiries)

        self.assertEqual(
            len(targets), stim_number,
            'Should have produced the correct number of targets for inquiries.')

        target_counts = np.zeros(stim_length + 1).astype(int)

        #count how many times each target position is used
        for pos in targets:
            target_counts[pos] = (target_counts[pos] + 1)

        #make sure position counts are equally distributed, and po
        for i in (target_counts[0:stim_length - 1]):
            self.assertTrue(i == 0)

        self.assertEqual(target_counts[stim_length], 100, 
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


if __name__ == '__main__':
    unittest.main()
