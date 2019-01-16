"""Tests for stimuli_generation helpers."""
import glob
import unittest
from os import path
import sounddevice as sd
import soundfile as sf
from mockito import any, mock, unstub, verify, when
from psychopy import core

from bcipy.helpers.stimuli_generation import play_sound, soundfiles,\
 random_rsvp_calibration_seq_gen, generate_icon_match_images, best_selection,\
 best_case_rsvp_seq_gen

MOCK_FS = 44100


class TestStimuliGeneration(unittest.TestCase):
    """This is Test Case for Stimuli Generation BCI data."""

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

        def mock_callback_function(timing):
            self.assertEqual(timing, self.test_timing)

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

        directory = "./sounds"
        soundfile_paths = [
            "./sounds/0.wav", "./sounds/1.wav", "./sounds/2.wav"
        ]
        when(glob).glob("./sounds/*.wav").thenReturn(soundfile_paths)
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
        directory = "./sounds/"
        soundfile_paths = [
            "./sounds/0.wav", "./sounds/1.wav", "./sounds/2.wav"
        ]
        when(glob).glob("./sounds/*.wav").thenReturn(soundfile_paths)
        when(path).isdir(directory).thenReturn(True)
        gen = soundfiles(directory)
        self.assertEqual(next(gen), soundfile_paths[0])

    def test_random_sequence_gen(self):
        """Test generation of random sequences"""
        alp = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '<', '_'
        ]
        num_sti = 10
        len_sti = 10
        seqs, seq_timings, seq_colors = random_rsvp_calibration_seq_gen(
            alp,
            timing=[0.5, 1, 0.2],
            color=['green', 'red', 'white'],
            num_sti=num_sti,
            len_sti=len_sti,
            is_txt=True)

        self.assertEqual(
            len(seqs), num_sti,
            "Should have produced the correct number of sequences")
        self.assertEqual(len(seq_timings), num_sti)
        self.assertEqual(len(seq_colors), num_sti)

        seq_strings = []
        for seq in seqs:
            self.assertEqual(
                len(seq), len_sti + 2,
                ("Sequence should include the correct number of choices as ",
                 "well as the target and cross."))
            choices = seq[2:]
            self.assertEqual(len_sti, len(set(choices)),
                             "All choices should be unique")

            # create a string of the options
            seq_strings.append(''.join(choices))

        self.assertEqual(
            len(seqs), len(set(seq_strings)),
            "All sequences should be different")


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
            "Included item should bump out the best item with the lowest val.")

        self.assertEqual(
            ['a', 'b', 'c'],
            best_selection(
                selection_elements=['a', 'b', 'c', 'd', 'e'],
                val=[0.5, 0.4, 0.1, 0.0, 0.0],
                len_query=3,
                always_included=['b']),
            "Included item should retain its position if it's already present")

        self.assertEqual(['a', 'b', 'e'],
                         best_selection(
                             selection_elements=['a', 'b', 'c', 'd', 'e'],
                             val=[0.5, 0.0, 0.1, 0.3, 0.0],
                             len_query=3,
                             always_included=['b', 'e']),
                         "multiple included items should be supported.")

        self.assertEqual(['d'],
                         best_selection(
                             selection_elements=['a', 'b', 'c', 'd', 'e'],
                             val=[0.5, 0.4, 0.1, 0.0, 0.0],
                             len_query=1,
                             always_included=['d', 'e']),
                         "len_query should be respected.")

        self.assertEqual(['a', 'b', 'c'],
                    best_selection(
                        selection_elements=['a', 'b', 'c', 'd', 'e'],
                        val=[0.5, 0.4, 0.1, 0.0, 0.0],
                        len_query=3,
                        always_included=['<']),
                    "should ignore items not in the set.")

    def test_best_case_sequence_gen(self):
        """Test best_case_rsvp_seq_gen"""
        alp = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        n = 5
        samples, times, colors = best_case_rsvp_seq_gen(
            alp=alp,
            session_stimuli=[0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2],
            timing=[1, 0.2],
            color=['red', 'white'],
            num_sti=1,
            len_sti=n,
            is_txt=True)

        first_seq = samples[0]
        self.assertEqual(1, len(samples))
        self.assertEqual(n + 1, len(first_seq),
                         "Should include fixation cross.")
        self.assertEqual(len(samples), len(times))
        self.assertEqual(len(samples), len(colors))

        expected = ['+', 'a', 'b', 'd', 'e', 'g']
        for letter in expected:
            self.assertTrue(letter in first_seq)

        self.assertNotEqual(expected, first_seq, "Should be in random order.")
        self.assertEqual([1] + ([0.2] * n), times[0])
        self.assertEqual(['red'] + (['white'] * n), colors[0])

    def test_best_case_sequence_gen_with_seq_constants(self):
        """Test best_case_rsvp_seq_gen with sequence constants"""

        alp = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        n = 5

        with self.assertRaises(
                Exception, msg="Constants should be in the alphabet"):
            best_case_rsvp_seq_gen(
                alp=alp,
                session_stimuli=[0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2],
                seq_constants=['<'])

        alp = ['a', 'b', 'c', 'd', 'e', 'f', 'g', '<']
        samples, times, colors = best_case_rsvp_seq_gen(
            alp=alp,
            session_stimuli=[0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2, 0.0],
            num_sti=1,
            len_sti=n,
            is_txt=True,
            seq_constants=['<'])

        first_seq = samples[0]
        self.assertEqual(1, len(samples))
        self.assertEqual(n + 1, len(first_seq),
                         "Should include fixation cross.")
        self.assertEqual(len(samples), len(times))
        self.assertEqual(len(samples), len(colors))

        expected = ['+', 'a', 'd', 'e', 'g', '<']
        for letter in expected:
            self.assertTrue(letter in first_seq)

        self.assertNotEqual(expected, first_seq, "Should be in random order.")
        self.assertEqual([1] + ([0.2] * n), times[0])
        self.assertEqual(['red'] + (['white'] * n), colors[0])


    def tearDown(self):
        unstub()


if __name__ == '__main__':
    unittest.main()
