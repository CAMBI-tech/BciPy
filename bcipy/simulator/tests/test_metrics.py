"""Test simulator metrics"""
import unittest
from unittest.mock import patch

from bcipy.simulator.metrics import (add_item, add_items, calculate_duration,
                                     get_final_typed, plot_save_path,
                                     rename_df_column, summarize)

SAMPLE_SERIES_DATA = {
    "1": {
        "0": {
            "next_display_state": "HELL"
        }
    },
    "2": {
        "0": {
            "next_display_state": "HELL"
        },
        "1": {
            "next_display_state": "HELLO"
        }
    }
}

SAMPLE_SESSION1 = {
    "series": SAMPLE_SERIES_DATA,
    "total_time_spent": 0.0,
    "total_minutes": 0.0,
    "total_number_series": 14,
    "total_inquiries": 45,
    "total_selections": 14,
    "inquiries_per_selection": 3.2,
    "task_summary": {
        "selections_correct": 14,
        "selections_incorrect": 0,
        "selections_correct_symbols": 14,
        "switch_total": 0,
        "switch_per_selection": 0.0,
        "switch_response_time": None,
        "typing_accuracy": 1.0,
        "correct_rate": 0,
        "copy_rate": 0
    }
}

SAMPLE_SESSION2 = {
    "series": SAMPLE_SERIES_DATA,
    "total_time_spent": 0.0,
    "total_minutes": 0.0,
    "total_number_series": 14,
    "total_inquiries": 16,
    "total_selections": 14,
    "inquiries_per_selection": 1.1428571428571428,
    "task_summary": {
        "selections_correct": 14,
        "selections_incorrect": 0,
        "selections_correct_symbols": 14,
        "switch_total": 0,
        "switch_per_selection": 0.0,
        "switch_response_time": None,
        "typing_accuracy": 1.0,
        "correct_rate": 0,
        "copy_rate": 0
    }
}


class TestSimMetrics(unittest.TestCase):
    """Tests for simulator metrics"""

    def test_add_item(self):
        """Test adding an item to a dict of sequences."""
        container = {'data1': [1]}
        add_item(container, 'data2', 10)
        self.assertTrue('data2' in container)
        self.assertEqual([10], container['data2'])

        add_item(container, 'data1', 2)
        self.assertEqual([1, 2], container['data1'])

    def test_get_final_typed(self):
        """Get final typed text from the series data"""
        session_data = {'series': SAMPLE_SERIES_DATA}
        self.assertEqual("HELLO", get_final_typed(session_data))

    def test_add_session_items(self):
        """Test session metrics accumulation"""
        combined = {}
        session_data = SAMPLE_SESSION1
        add_items(combined, session_data, inquiry_seconds=2)
        for key in [
                "total_number_series", "total_inquiries", "total_selections",
                "inquiries_per_selection", "task_summary__selections_correct",
                "task_summary__selections_incorrect",
                "task_summary__selections_correct_symbols",
                "task_summary__typing_accuracy", "typed"
        ]:
            self.assertTrue(key in combined)
            self.assertEqual(1, len(combined[key]))

    @patch('bcipy.simulator.metrics.max_inquiry_duration')
    @patch('bcipy.simulator.metrics.sim_parameters')
    @patch('bcipy.simulator.metrics.load')
    @patch('bcipy.simulator.metrics.open')
    @patch('bcipy.simulator.metrics.session_paths')
    def test_summarize(self, session_path_mock, open_mock, json_mock,
                       sim_parameters_mock, max_inquiry_duration_mock):
        """Test summary function"""
        session_path_mock.return_value = [
            'run1/session.json', 'run2/session.json'
        ]
        json_mock.side_effect = [SAMPLE_SESSION1, SAMPLE_SESSION2]
        max_inquiry_duration_mock.return_value = 2
        result = summarize("sim-dir")

        sim_parameters_mock.assert_called_once_with("sim-dir")
        max_inquiry_duration_mock.assert_called_once()
        self.assertEqual(open_mock.call_count, 2)
        for key in [
                "total_number_series", "total_inquiries", "total_selections",
                "inquiries_per_selection", "task_summary__selections_correct",
                "task_summary__selections_incorrect",
                "task_summary__selections_correct_symbols",
                "task_summary__typing_accuracy", "typed"
        ]:
            self.assertTrue(key in result)
            self.assertEqual(2, len(result[key]))

    def test_rename_df_column(self):
        """Test renaming columns for output"""
        self.assertEqual("total_inquiries",
                         rename_df_column("total_inquiries"))
        self.assertEqual("selections_correct",
                         rename_df_column("task_summary__selections_correct"))

    def test_plot_name(self):
        """Test naming for plots"""
        self.assertEqual("sim_dir/metrics.png", plot_save_path("sim_dir"))

    def test_compute_duration(self):
        """Test the calculation of the overall session duration"""
        self.assertEqual(
            250, calculate_duration(inquiry_count=100, inquiry_seconds=2.5))


if __name__ == '__main__':
    unittest.main()
