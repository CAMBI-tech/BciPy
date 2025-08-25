import glob
import unittest
from collections import Counter
from os import path

import numpy as np
import sounddevice as sd
import soundfile as sf
from mockito import any, mock, unstub, verify, when
from psychopy import core

from bcipy.core.stimuli import (DEFAULT_FIXATION_PATH, InquiryReshaper,
                                StimuliOrder, TargetPositions, TrialReshaper,
                                alphabetize, best_case_rsvp_inq_gen,
                                best_selection, distributed_target_positions,
                                generate_calibration_inquiries,
                                generate_inquiry, generate_targets,
                                get_fixation, inquiry_nontarget_counts,
                                inquiry_target, inquiry_target_counts,
                                jittered_timing, play_sound,
                                random_target_positions, soundfiles,
                                target_index, update_inquiry_timing)
from bcipy.exceptions import BciPyCoreException

MOCK_FS = 44100


def is_uniform_distribution(inquiry_count: int, stim_per_inquiry: int,
                            percentage_without_target: int,
                            counter: Counter) -> bool:
    """Determine if the counts in the provided counter are distributed
    uniformly."""

    no_target_inquiries = (int)(inquiry_count *
                                (percentage_without_target / 100))
    target_inquiries = inquiry_count - no_target_inquiries
    expected_targets_per_position = (int)(target_inquiries / stim_per_inquiry)

    return all(expected_targets_per_position <= counter[pos] <=
               expected_targets_per_position + 1 for pos in counter
               if pos is not None)


class TestStimuliGeneration(unittest.TestCase):
    """This is Test Case for Stimuli Generated via BciPy."""

    alp = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '<', '_'
    ]

    def tearDown(self):
        unstub()

    def test_calibration_inquiry_generator_with_jitter(self):
        inquiry_count = 10
        stim_per_inquiry = 10
        stim_timing = [0.5, 1, 2]
        stim_jitter = 1

        max_jitter = stim_timing[-1] + stim_jitter
        min_jitter = stim_timing[-1] - stim_jitter
        inquiries, inq_timings, inq_colors = generate_calibration_inquiries(
            self.alp,
            timing=stim_timing,
            inquiry_count=inquiry_count,
            stim_per_inquiry=stim_per_inquiry,
            jitter=stim_jitter)

        self.assertEqual(
            len(inquiries), inquiry_count,
            'Should have produced the correct number of inquiries')
        self.assertEqual(len(inq_timings), inquiry_count)
        self.assertEqual(len(inq_colors), inquiry_count)

        inq_strings = []
        for inq in inquiries:
            self.assertEqual(
                len(inq), stim_per_inquiry + 2,
                ('inquiry should include the correct number of choices as ',
                 'well as the target and cross.'))
            choices = inq[2:]
            self.assertEqual(stim_per_inquiry, len(set(choices)),
                             'All choices should be unique')

            # create a string of the options
            inq_strings.append(''.join(choices))

        # ensure timing is jittered
        for j in inq_timings:
            inq_timing = j[2:]  # remove the target presentaion and cross
            for inq_time in inq_timing:
                self.assertTrue(
                    min_jitter <= inq_time <= max_jitter,
                    'Timing should be jittered and within the correct range')

            self.assertTrue(
                len(set(inq_timing)) > 1, 'All choices should be unique')

        self.assertEqual(len(inquiries), len(set(inq_strings)),
                         'All inquiries should be different')

    def test_calibration_inquiry_generator_random_order(self):
        """Test generation of random inquiries"""
        inquiry_count = 10
        stim_per_inquiry = 10
        inquiries, inq_timings, inq_colors = generate_calibration_inquiries(
            self.alp,
            inquiry_count=inquiry_count,
            stim_per_inquiry=stim_per_inquiry,
            stim_order=StimuliOrder.RANDOM)

        self.assertEqual(
            len(inquiries), inquiry_count,
            'Should have produced the correct number of inquiries')
        self.assertEqual(len(inq_timings), inquiry_count)
        self.assertEqual(len(inq_colors), inquiry_count)

        inq_strings = []
        for inq in inquiries:
            self.assertEqual(
                len(inq), stim_per_inquiry + 2,
                ('inquiry should include the correct number of choices as ',
                 'well as the target and cross.'))
            choices = inq[2:]
            self.assertEqual(stim_per_inquiry, len(set(choices)),
                             'All choices should be unique')

            # create a string of the options
            inq_strings.append(''.join(choices))

        self.assertEqual(len(inquiries), len(set(inq_strings)),
                         'All inquiries should be different')

    def test_calibration_inquiry_generator_alphabetical_order(self):
        """Test generation of random inquiries"""
        inquiry_count = 10
        stim_per_inquiry = 10
        inquiries, inq_timings, inq_colors = generate_calibration_inquiries(
            self.alp,
            inquiry_count=inquiry_count,
            stim_per_inquiry=stim_per_inquiry,
            stim_order=StimuliOrder.ALPHABETICAL)

        self.assertEqual(
            len(inquiries), inquiry_count,
            'Should have produced the correct number of inquiries')
        self.assertEqual(len(inq_timings), inquiry_count)
        self.assertEqual(len(inq_colors), inquiry_count)

        inq_strings = []
        for inq in inquiries:
            self.assertEqual(
                len(inq), stim_per_inquiry + 2,
                ('inquiry should include the correct number of choices as ',
                 'well as the target and cross.'))
            choices = inq[2:]
            self.assertEqual(stim_per_inquiry, len(set(choices)),
                             'All choices should be unique')
            self.assertEqual(alphabetize(choices), choices)

            # create a string of the options
            inq_strings.append(''.join(choices))

        self.assertEqual(len(inquiries), len(set(inq_strings)),
                         'All inquiries should be different')

    def test_target_index(self):
        """Test target_index function"""
        inquiry = ['T', '+', 'G', 'J', 'K', 'L', 'M', 'Q', 'T', 'V', 'X', '<']
        self.assertEqual(target_index(inquiry), 6)

        inquiry = ['A', '+', 'G', 'J', 'K', 'L', 'M', 'Q', 'T', 'V', 'X', '<']
        self.assertEqual(target_index(inquiry), None)

    def test_calibration_inquiry_generator_distributed_targets(self):
        """Test generation of inquiries with distributed target positions"""
        inquiry_count = 100
        stim_per_inquiry = 10
        percentage_without_target = 10
        inquiries, inq_timings, inq_colors = generate_calibration_inquiries(
            self.alp,
            inquiry_count=inquiry_count,
            stim_per_inquiry=stim_per_inquiry,
            stim_order=StimuliOrder.RANDOM,
            target_positions=TargetPositions.DISTRIBUTED,
            percentage_without_target=percentage_without_target)

        self.assertEqual(
            len(inquiries), inquiry_count,
            'Should have produced the correct number of inquiries')
        self.assertEqual(len(inq_timings), inquiry_count)
        self.assertEqual(len(inq_colors), inquiry_count)

        inq_strings = []
        for inq in inquiries:
            self.assertEqual(
                len(inq), stim_per_inquiry + 2,
                ('inquiry should include the correct number of choices as ',
                 'well as the target and cross.'))
            choices = inq[2:]
            self.assertEqual(stim_per_inquiry, len(set(choices)),
                             'All choices should be unique')

            # create a string of the options
            inq_strings.append(''.join(choices))

        self.assertEqual(len(inquiries), len(set(inq_strings)),
                         'All inquiries should be different')

        # Test distribution
        counter = Counter(target_index(inq) for inq in inquiries)
        self.assertTrue(
            is_uniform_distribution(inquiry_count, stim_per_inquiry,
                                    percentage_without_target, counter))

    def test_calibration_inquiry_generator_distributed_targets_alphabetical(
            self):
        """Test generation of inquiries with distributed target positions"""
        inquiry_count = 100
        stim_per_inquiry = 10
        percentage_without_target = 20
        inquiries, inq_timings, inq_colors = generate_calibration_inquiries(
            self.alp,
            inquiry_count=inquiry_count,
            stim_per_inquiry=stim_per_inquiry,
            stim_order=StimuliOrder.ALPHABETICAL,
            target_positions=TargetPositions.DISTRIBUTED,
            percentage_without_target=percentage_without_target)

        self.assertEqual(
            len(inquiries), inquiry_count,
            'Should have produced the correct number of inquiries')
        self.assertEqual(len(inq_timings), inquiry_count)
        self.assertEqual(len(inq_colors), inquiry_count)

        inq_strings = []
        for inq in inquiries:
            self.assertEqual(
                len(inq), stim_per_inquiry + 2,
                ('inquiry should include the correct number of choices as ',
                 'well as the target and cross.'))
            choices = inq[2:]
            self.assertEqual(stim_per_inquiry, len(set(choices)),
                             'All choices should be unique')

            # create a string of the options
            inq_strings.append(''.join(choices))

        self.assertEqual(len(inquiries), len(set(inq_strings)),
                         'All inquiries should be different')

    def test_calibration_inquiry_generator_distributed_targets_no_nontargets(
            self):
        """Test generation of inquiries with distributed target positions and no nontarget inquiries."""
        inquiry_count = 100
        stim_per_inquiry = 10
        percentage_without_target = 0
        inquiries, inq_timings, inq_colors = generate_calibration_inquiries(
            self.alp,
            inquiry_count=inquiry_count,
            stim_per_inquiry=stim_per_inquiry,
            stim_order=StimuliOrder.RANDOM,
            target_positions=TargetPositions.DISTRIBUTED,
            percentage_without_target=percentage_without_target)

        self.assertEqual(
            len(inquiries), inquiry_count,
            'Should have produced the correct number of inquiries')
        self.assertEqual(len(inq_timings), inquiry_count)
        self.assertEqual(len(inq_colors), inquiry_count)

        inq_strings = []
        for inq in inquiries:
            self.assertEqual(
                len(inq), stim_per_inquiry + 2,
                ('inquiry should include the correct number of choices as ',
                 'well as the target and cross.'))
            choices = inq[2:]
            self.assertEqual(stim_per_inquiry, len(set(choices)),
                             'All choices should be unique')

            # create a string of the options
            inq_strings.append(''.join(choices))

        self.assertEqual(len(inquiries), len(set(inq_strings)),
                         'All inquiries should be different')

        # Ensure all inquiries include the target
        for inq in inquiries:
            self.assertIsNotNone(target_index(inq))

    def test_calibration_inquiry_generator_distributed_targets_positions(self):
        """Test generation of distributed target positions with nontarget inquiries."""

        inquiry_count = 11
        stim_per_inquiry = 10
        percentage_without_target = 10

        target_positions = distributed_target_positions(
            inquiry_count=inquiry_count,
            stim_per_inquiry=stim_per_inquiry,
            percentage_without_target=percentage_without_target)

        self.assertEqual(
            len(target_positions), inquiry_count,
            'Should have produced the correct number of target_positions for inquiries.'
        )

        self.assertTrue(
            is_uniform_distribution(inquiry_count, stim_per_inquiry,
                                    percentage_without_target,
                                    Counter(target_positions)))

    def test_calibration_inquiry_generator_distributed_targets_positions_half_nontarget(
            self):
        """Test generation of distributed target positions with half being nontarget inquiries."""

        inquiry_count = 120
        stim_per_inquiry = 9
        percentage_without_target = 50

        nontarget_inquiries = (int)(inquiry_count *
                                    (percentage_without_target / 100))

        target_positions = distributed_target_positions(
            inquiry_count=inquiry_count,
            stim_per_inquiry=stim_per_inquiry,
            percentage_without_target=percentage_without_target)

        self.assertEqual(
            len(target_positions), inquiry_count,
            'Should have produced the correct number of target_positions for inquiries.'
        )
        # count how many times each target position is used
        counter = Counter(target_positions)

        # make sure correct number of non-target inquiries
        self.assertEqual(
            counter[None], nontarget_inquiries,
            'Should have produced 50 percent of 120 non-target positions.')

        self.assertTrue(
            is_uniform_distribution(inquiry_count, stim_per_inquiry,
                                    percentage_without_target, counter))

    def test_calibration_inquiry_generator_distributed_targets_positions_no_nontargets(
            self):
        """Test generation of distributed target positions with no nontarget inquiries."""

        inquiry_count = 50
        stim_per_inquiry = 11
        percentage_without_target = 0

        target_positions = distributed_target_positions(
            inquiry_count=inquiry_count,
            stim_per_inquiry=stim_per_inquiry,
            percentage_without_target=percentage_without_target)

        self.assertEqual(
            len(target_positions), inquiry_count,
            'Should have produced the correct number of target_positions for inquiries.'
        )

        # count how many times each target position is used
        counter = Counter(target_positions)

        self.assertTrue(
            is_uniform_distribution(inquiry_count, stim_per_inquiry,
                                    percentage_without_target, counter))

        # make sure there are no non-target inquiries
        self.assertEqual(counter[None], 0,
                         'Should have produced no non-target positions.')

    def test_calibration_inquiry_generator_distributed_targets_all_nontargets(
            self):
        """Test generation of distributed target positions with all inquiries being non-target."""

        inquiry_count = 100
        stim_per_inquiry = 6
        percentage_without_target = 100

        target_positions = distributed_target_positions(
            inquiry_count=inquiry_count,
            stim_per_inquiry=stim_per_inquiry,
            percentage_without_target=percentage_without_target)

        self.assertEqual(
            len(target_positions), inquiry_count,
            'Should have produced the correct number of target_positions for inquiries.'
        )

        # count how many times each target position is used
        counter = Counter(target_positions)

        self.assertTrue(
            is_uniform_distribution(inquiry_count, stim_per_inquiry,
                                    percentage_without_target, counter))

        # make sure all inquries are non-target inquiries
        self.assertEqual(counter[None], inquiry_count,
                         'Should have produced all non-target positions.')

    def test_random_target_positions(self):
        """Test generation of random target positions with nontarget inquiries."""

        inquiry_count = 100
        stim_per_inquiry = 10
        percentage_without_target = 10

        target_positions = random_target_positions(
            inquiry_count=inquiry_count,
            stim_per_inquiry=stim_per_inquiry,
            percentage_without_target=percentage_without_target)

        self.assertEqual(
            len(target_positions), inquiry_count,
            'Should have produced the correct number of target_positions for inquiries.'
        )
        counter = Counter(target_positions)
        print(counter)
        self.assertFalse(
            is_uniform_distribution(inquiry_count, stim_per_inquiry,
                                    percentage_without_target,
                                    counter))

    def test_random_target_positions_half_nontarget(self):
        """Test generation of random target positions with half being nontarget inquiries."""

        inquiry_count = 120
        stim_per_inquiry = 9
        percentage_without_target = 50

        target_positions = random_target_positions(
            inquiry_count=inquiry_count,
            stim_per_inquiry=stim_per_inquiry,
            percentage_without_target=percentage_without_target)

        self.assertEqual(
            len(target_positions), inquiry_count,
            'Should have produced the correct number of target_positions for inquiries.'
        )
        # count how many times each target position is used
        counter = Counter(target_positions)

        # make sure correct number of non-target inquiries
        self.assertEqual(
            counter[None], 60,
            'Should have produced 50 percent of 120 non-target positions.')

        self.assertFalse(
            is_uniform_distribution(inquiry_count, stim_per_inquiry,
                                    percentage_without_target, counter))

    def test_random_target_positions_no_nontargets(self):
        """Test generation of random target positions with no nontarget inquiries."""

        inquiry_count = 50
        stim_per_inquiry = 11
        percentage_without_target = 0

        target_positions = random_target_positions(
            inquiry_count=inquiry_count,
            stim_per_inquiry=stim_per_inquiry,
            percentage_without_target=percentage_without_target)

        self.assertEqual(
            len(target_positions), inquiry_count,
            'Should have produced the correct number of target_positions for inquiries.'
        )

        # count how many times each target position is used
        counter = Counter(target_positions)

        self.assertFalse(
            is_uniform_distribution(inquiry_count, stim_per_inquiry,
                                    percentage_without_target, counter))

        # make sure there are no non-target inquiries
        self.assertEqual(counter[None], 0,
                         'Should have produced no non-target positions.')

    def test_random_target_positions_all_nontargets(self):
        """Test generation of random target positions with all inquiries being non-target."""
        inquiry_count = 100

        target_positions = random_target_positions(
            inquiry_count=inquiry_count,
            stim_per_inquiry=6,
            percentage_without_target=100)

        # count how many times each target position is used
        counter = Counter(target_positions)

        # make sure all inquries are non-target inquiries
        self.assertEqual(counter[None], inquiry_count,
                         'Should have produced all non-target positions.')

    def test_generate_targets(self):
        """Test target generation"""
        symbols = ['A', 'B', 'C', 'D']
        targets = generate_targets(
            symbols, inquiry_count=9, percentage_without_target=0)
        self.assertEqual(len(targets), 8)
        self.assertTrue(all(val == 2 for val in Counter(targets).values()))

        targets = generate_targets(
            symbols, inquiry_count=9, percentage_without_target=50)
        self.assertEqual(len(targets), 4)
        self.assertTrue(all(val == 1 for val in Counter(targets).values()))

    def test_inquiry_target(self):
        """Test inquiry target behavior."""
        symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        inquiry = ['C', 'A', 'F', 'E']
        next_targets = ['Q', 'F', 'A']
        target = inquiry_target(inquiry,
                                target_position=0,
                                symbols=symbols,
                                next_targets=next_targets)

        self.assertEqual(
            target, 'F',
            'the inquiry symbol next in order in the target choices should have been selected'
        )
        self.assertSequenceEqual(inquiry, ['F', 'A', 'C', 'E'],
                                 'inquiry should have been re-ordered')
        self.assertSequenceEqual(
            next_targets, ['Q', 'A'],
            'list of next targets should have been modified')

    def test_inquiry_target_with_none_position(self):
        """Test inquiry target with position of None."""
        symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        inquiry = ['C', 'A', 'F', 'E']
        next_targets = ['Q', 'F', 'A']
        target = inquiry_target(inquiry, None, symbols, next_targets)
        self.assertTrue(target not in inquiry)
        self.assertTrue(target in symbols)
        self.assertSequenceEqual(
            inquiry, ['C', 'A', 'F', 'E'], 'inquiry should not have changed')

    def test_inquiry_target_missing(self):
        """Test inquiry target where none of the next_targets are present in
        the current inquiry."""
        symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        inquiry = ['C', 'A', 'F', 'E']
        next_targets = ['Q', 'D']
        target = inquiry_target(inquiry,
                                target_position=0,
                                symbols=symbols,
                                next_targets=next_targets)
        self.assertTrue(target in inquiry)
        self.assertTrue(target not in next_targets)
        self.assertSequenceEqual(
            inquiry, ['C', 'A', 'F', 'E'], 'inquiry should not have changed')
        self.assertSequenceEqual(
            next_targets, ['Q', 'D'], 'next_targets should not have changed')

    def test_inquiry_target_no_targets(self):
        """Test inquiry_target when no next_targets are provided"""
        symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        inquiry = ['C', 'A', 'F', 'E']
        next_targets = ['Q', 'D']
        target = inquiry_target(inquiry,
                                target_position=0,
                                symbols=symbols,
                                next_targets=next_targets)
        self.assertEqual(
            target, 'C', 'should have used the target_position to get the target')
        self.assertSequenceEqual(
            inquiry, ['C', 'A', 'F', 'E'], 'inquiry should not have changed')

    def test_inquiry_target_last_target(self):
        """Test inquiry target behavior."""
        symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        inquiry = ['C', 'A', 'F', 'E']
        next_targets = ['Q', 'F', 'A']
        target = inquiry_target(inquiry,
                                target_position=0,
                                symbols=symbols,
                                next_targets=next_targets,
                                last_target='F')

        self.assertEqual(target, 'A', 'last_target should have been skipped')
        self.assertSequenceEqual(inquiry, ['A', 'C', 'F', 'E'],
                                 'inquiry should have been re-ordered')
        self.assertSequenceEqual(
            next_targets, ['Q', 'F'],
            'list of next targets should have been modified')

    def test_generate_inquiry(self):
        """Test generation of a single inquiry"""
        symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        inquiry = generate_inquiry(symbols=symbols,
                                   length=3,
                                   stim_order=StimuliOrder.RANDOM)

        # NOTE: it's possible an inquiry in Random order will be in alphabetical order
        self.assertEqual(3, len(inquiry))
        for sym in inquiry:
            self.assertTrue(sym in symbols)

        inquiry = generate_inquiry(symbols=symbols,
                                   length=3,
                                   stim_order=StimuliOrder.ALPHABETICAL)
        self.assertEqual(inquiry, sorted(inquiry))

    def test_inquiry_target_counts(self):
        """Test inquiry_target_counts"""
        inquiries = [['D', '+', 'Z', 'Q', 'T', 'C', 'D'],
                     ['Q', '+', 'L', 'Q', 'B', 'J', 'N'],
                     ['W', '+', 'Q', 'F', 'M', 'W', 'N'],
                     ['Q', '+', 'X', 'G', 'Z', 'J', 'V']]

        targeted = ['D', 'Q', 'W']
        not_targeted = list(set(self.alp) - set(targeted))
        target_counts = inquiry_target_counts(inquiries, symbols=self.alp)
        self.assertTrue(all(target_counts[sym] == 1 for sym in targeted))
        self.assertTrue(all(target_counts[sym] == 0 for sym in not_targeted))

    def test_inquiry_nontarget_counts(self):
        """Test inquiry_nontarget_counts"""
        symbols = ['A', 'B', 'C', 'D', 'E', 'F']
        inquiries = [['C', '+', 'B', 'E', 'C'],
                     ['E', '+', 'B', 'F', 'E'],
                     ['E', '+', 'A', 'C', 'E'],
                     ['B', '+', 'D', 'E', 'B'],
                     ['C', '+', 'F', 'B', 'C']]

        expected = {'A': 1, 'B': 3, 'C': 1, 'D': 1, 'E': 2, 'F': 2}
        counts = inquiry_nontarget_counts(inquiries, symbols=symbols)
        self.assertDictEqual(counts, expected)

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
            best_selection(selection_elements=['a', 'b', 'c', 'd', 'e'],
                           val=[0.3, 0.1, 0.3, 0.1, 0.2],
                           len_query=3,
                           always_included=['d']),
            'Included item should bump out the best item with the lowest val.')

        self.assertEqual(
            ['a', 'b', 'c'],
            best_selection(selection_elements=['a', 'b', 'c', 'd', 'e'],
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

    def test_best_case_inquiry_gen_invalid_alp(self):
        """Test best_case_rsvp_inq_gen throws error when passed invalid alp shape"""
        alp = ['a', 'b', 'c', 'd']
        session_stimuli = [0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2]
        stim_length = 5
        with self.assertRaises(BciPyCoreException, msg='Missing information about the alphabet.'):
            best_case_rsvp_inq_gen(
                alp=alp,
                session_stimuli=session_stimuli,
                timing=[1, 0.2],
                color=['red', 'white'],
                stim_number=1,
                stim_length=stim_length,
                is_txt=True
            )

    def test_best_case_inquiry_gen_with_inq_constants(self):
        """Test best_case_rsvp_inq_gen with inquiry constants"""
        alp = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        n = 5

        with self.assertRaises(Exception,
                               msg='Constants should be in the alphabet'):
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

        self.assertEqual([1] + ([0.2] * n), times[0])
        self.assertEqual(['red'] + (['white'] * n), colors[0])

    def test_best_case_inq_gen_is_random(self):
        """Test that best_case_inq_gen produces random results. While this test can technically fail,
        the odds are incredibly low, around 1 in 1 million"""
        samps = set()
        for i in range(25):
            alp = ['a', 'b', 'c', 'd', 'e', 'f', 'g', '<']
            samples, _, _ = best_case_rsvp_inq_gen(
                alp=alp,
                session_stimuli=[0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2, 0.0],
                stim_number=1,
                stim_length=8,
                is_txt=True,
                inq_constants=['<'])
            samps.add(tuple(samples[0]))
        self.assertTrue(
            len(samps) > 1, '`best_case_rsvp_inq_gen` Should produce random results')


class TestJitteredTiming(unittest.TestCase):

    def test_jittered_timing_returns_correct_number_of_stims(self):
        time = 1
        jitter = 0.24
        stim_number = 10

        resp = jittered_timing(time, jitter, stim_number)

        self.assertEqual(stim_number, len(resp))
        self.assertIsInstance(resp, list)

    def test_jittered_timing_with_jitter_within_defined_limits(self):
        time = 1
        jitter = 0.25
        stim_number = 100

        max_jitter = time + jitter
        min_jitter = time - jitter

        resp = jittered_timing(time, jitter, stim_number)

        for r in resp:
            self.assertTrue(min_jitter <= r <= max_jitter)

    def test_jittered_timing_throw_exception_when_jitter_greater_than_time(
            self):
        # to prevent 0 values we prevent the jitter from being greater than the time
        time = 1
        jitter = 1.5
        stim_number = 100

        with self.assertRaises(Exception,
                               msg='Jitter should be less than stimuli time'):
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
        stimuli = ['Z', character, 'Q', 'A', 'G']
        expected = ['A', 'G', 'Q', 'Z', character]
        response = alphabetize(stimuli)
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
        expected_shape = (self.channel_number, len(self.target_info),
                          trial_length_samples)
        self.assertTrue(np.all(labels == [1, 0, 0]))
        self.assertTrue(reshaped_trials.shape == expected_shape)

    def test_trial_reshaper_with_no_channel_map(self):
        sample_rate = 256
        trial_length_s = 0.5
        reshaped_trials, labels = TrialReshaper()(
            trial_targetness_label=self.target_info,
            timing_info=self.timing_info,
            eeg_data=self.eeg,
            sample_rate=sample_rate,
            channel_map=None,
            poststimulus_length=trial_length_s
        )
        trial_length_samples = int(sample_rate * trial_length_s)
        expected_shape = (self.channel_number, len(
            self.target_info), trial_length_samples)
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
            "target",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "nontarget",
            "target",
            "nontarget",
            "nontarget",
            "nontarget",
            "target",
        ]
        self.true_labels = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0],
                                     [0, 0, 1]])
        self.timing_info = [
            1.4,
            2.4,
            3.4,
            28.1,
            30.1,
            38.1,
            50.1,
            51.1,
            52.1,
            80.1,
            81.1,
            82.1,
        ]
        # Note this value must be greater than the difference between first and last timing info per inquiry
        # In this case there are 3 trials per inquiry and the difference between the first and last timing info is 10s
        self.inquiry_duration_s = 10 + self.trial_length
        self.samples_per_inquiry = int(self.sample_rate *
                                       self.inquiry_duration_s)
        self.samples_per_trial = int(self.sample_rate * self.trial_length)
        # We create a wealth of samples, but only use the first 4 inquiries worth
        self.eeg = np.random.randn(self.n_channel, 10000)
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
        expected_shape = (self.n_channel, self.n_inquiry,
                          self.samples_per_inquiry)
        self.assertTrue(reshaped_data.shape == expected_shape)
        self.assertTrue(np.all(labels == self.true_labels))

    def test_inquiry_reshaper_with_no_channel_map(self):
        reshaped_data, labels, _ = InquiryReshaper()(
            trial_targetness_label=self.target_info,
            timing_info=self.timing_info,
            eeg_data=self.eeg,
            sample_rate=self.sample_rate,
            trials_per_inquiry=self.trials_per_inquiry,
            channel_map=None,
            poststimulus_length=self.trial_length
        )
        expected_shape = (self.n_channel, self.n_inquiry,
                          self.samples_per_inquiry)
        self.assertTrue(reshaped_data.shape == expected_shape)
        self.assertTrue(np.all(labels == self.true_labels))

    def test_inquiry_reshaper_trial_extraction(self):
        timing = [[1, 3, 4], [1, 4, 5], [1, 2, 3], [4, 5, 6]]
        # make a fake eeg data array (n_channels, n_inquiry, n_samples)
        inquiries = np.random.randn(self.n_channel, self.n_inquiry,
                                    self.samples_per_inquiry)

        response = InquiryReshaper().extract_trials(
            inquiries=inquiries,
            samples_per_trial=self.samples_per_trial,
            inquiry_timing=timing,
        )
        expected_shape = (self.n_channel,
                          (self.trials_per_inquiry * self.n_inquiry),
                          self.samples_per_trial)
        self.assertTrue(response.shape == expected_shape)

    def test_inquiry_reshaper_trial_extraction_with_prestimulus(self):
        timing = [[2, 7, 10], [2, 7, 15], [2, 12, 15], [4, 5, 6]]
        # make a fake eeg data array (n_channels, n_inquiry, n_samples)
        inquiries = np.random.randn(self.n_channel, self.n_inquiry,
                                    self.samples_per_inquiry)
        prestimulus_samples = 1
        response = InquiryReshaper().extract_trials(
            inquiries=inquiries,
            samples_per_trial=self.samples_per_trial,
            inquiry_timing=timing,
            prestimulus_samples=prestimulus_samples,
        )
        expected_shape = (self.n_channel,
                          (self.trials_per_inquiry * self.n_inquiry),
                          (self.samples_per_trial + prestimulus_samples))
        self.assertTrue(response.shape == expected_shape)


class TestUpdateInquiryTiming(unittest.TestCase):

    def test_update_inquiry_timing(self):
        initial_timing = [[100, 200]]
        downsample = 2
        new_timing = update_inquiry_timing(
            timing=initial_timing,
            downsample=downsample,
        )
        expected_timing = [[50, 100]]
        self.assertTrue(new_timing == expected_timing)

    def test_update_inquiry_timing_with_non_int_timing_after_correction(self):
        initial_timing = [[100, 201]]
        downsample = 2
        new_timing = update_inquiry_timing(
            timing=initial_timing,
            downsample=downsample,
        )
        expected_timing = [[50, 201 // downsample]]
        self.assertTrue(new_timing == expected_timing)


class TestSoundStimuli(unittest.TestCase):

    def tearDown(self):
        unstub()

    def test_play_sound_returns_timing(self):
        # fake sound file path
        sound_file_path = 'test_sound_file_path'

        # mock the other library interactions
        when(sf).read(sound_file_path, dtype='float32').thenReturn(
            ('data', MOCK_FS))
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
        when(sf).read(sound_file_path,
                      dtype='float32').thenRaise(Exception(''))

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
        when(sf).read(sound_file_path, dtype='float32').thenReturn(
            ('data', MOCK_FS))
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
        when(glob).glob(path.join(directory,
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
        when(glob).glob(path.join(directory,
                                  '*.wav')).thenReturn(soundfile_paths)
        when(path).isdir(directory).thenReturn(True)
        gen = soundfiles(directory)
        self.assertEqual(next(gen), soundfile_paths[0])


if __name__ == '__main__':
    unittest.main()
