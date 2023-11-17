import unittest

from bcipy.helpers.symbols import alphabet
from bcipy.task.paradigm.vep.stim_generation import (
    generate_vep_calibration_inquiries, generate_vep_inquiry, stim_per_box)


class SSVEPStimuli(unittest.TestCase):
    """Tests for VEP inquiry generation"""

    def test_stim_per_box(self):
        """Test number of stimuli per box"""
        symbol_count = 27
        for _i in range(10):
            counts = stim_per_box(num_symbols=symbol_count,
                                  num_boxes=6,
                                  max_empty_boxes=1,
                                  max_single_sym_boxes=4)
            self.assertEqual(symbol_count, sum(counts))
            self.assertTrue(
                sum([cnt for cnt in counts if cnt == 1]) <= 4,
                "There should be no more than 4 single-symbol boxes")
            self.assertTrue(
                sum([cnt for cnt in counts if cnt == 0]) <= 1,
                "There should be no more than 1 empty box")

    def test_stim_per_box_few_boxes(self):
        """Test number of stimuli per box when max values sum to more than the
         number of boxes."""
        symbol_count = 27
        for _i in range(10):
            counts = stim_per_box(num_symbols=symbol_count,
                                  num_boxes=4,
                                  max_empty_boxes=1,
                                  max_single_sym_boxes=4)
            self.assertEqual(symbol_count, sum(counts))
            self.assertTrue(
                sum([cnt for cnt in counts if cnt == 1]) <= 3,
                "There should be no more than 3 single-symbol boxes")
            self.assertTrue(
                sum([cnt for cnt in counts if cnt == 0]) == 0,
                "There should be no empty boxes")

    def test_stim_per_box_few_boxes_adjusted_params(self):
        """Test number of stimuli per box where max values have been
        adjusted."""
        symbol_count = 27
        for _i in range(10):
            counts = stim_per_box(num_symbols=symbol_count,
                                  num_boxes=4,
                                  max_empty_boxes=1,
                                  max_single_sym_boxes=2)
            self.assertEqual(symbol_count, sum(counts))
            self.assertTrue(
                sum([cnt for cnt in counts if cnt == 1]) <= 2,
                "There should be no more than 2 single-symbol boxes")
            self.assertTrue(
                sum([cnt for cnt in counts if cnt == 0]) <= 1,
                "There should be at most one empty box")

    def test_generate_single_vep_calib_inquiry(self):
        """Test generation of a single VEP calibration inquiry"""
        symbols = alphabet()
        box_count = 6
        inq = generate_vep_inquiry(symbols,
                                   num_boxes=box_count,
                                   target='M',
                                   target_pos=2)

        self.assertEqual(box_count, len(inq))
        self.assertTrue('M' in inq[2])
        self.assertEqual(len(symbols),
                         sum([len(box_items) for box_items in inq]))

    def test_generate_vep_calibration_inquiries(self):
        """Test generate calibration inquiries"""
        alp = alphabet()
        stim_colors = ['green', 'red', 'blue', 'yellow', 'orange', 'purple']
        stim_timing = [4, 0.5, 4]
        schedule = generate_vep_calibration_inquiries(
            alp=alp,
            timing=stim_timing,
            color=stim_colors,
            inquiry_count=20,
            num_boxes=4)

        self.assertEqual(20, len(schedule.stimuli))
        self.assertEqual(6, len(schedule.stimuli[0]))
        for inq in schedule.stimuli:
            self.assertEqual(6, len(inq))
            self.assertTrue(isinstance(inq[0], str),
                            "First item should be the target")
            self.assertTrue(isinstance(inq[1], str),
                            "Second item should be the fixation")
            for pos in range(2, 6):
                self.assertTrue(isinstance(inq[pos], list),
                                "Item should be a list of symbols")

        for color in schedule.colors:
            self.assertEqual(stim_colors, color)
        for timing in schedule.durations:
            self.assertEqual(stim_timing, timing)


if __name__ == '__main__':
    unittest.main()
