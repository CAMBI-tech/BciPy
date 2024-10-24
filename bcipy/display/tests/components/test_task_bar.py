"""Task bar component tests"""
import unittest
from unittest.mock import patch, Mock
from bcipy.display.components.task_bar import TaskBar, CalibrationTaskBar, CopyPhraseTaskBar


class TestTaskBar(unittest.TestCase):
    """Test task bar components."""
    @patch('bcipy.display.components.task_bar.visual')
    def test_task_bar(self, visual_mock):
        """Test Task Bar"""
        text_stim_mock = Mock()
        border_mock = Mock()

        visual_mock.TextStim.return_value = text_stim_mock
        visual_mock.Line.return_value = border_mock

        win = Mock(size=(500, 500), units="norm")
        task_bar = TaskBar(win, text='Testing')
        self.assertEqual(len(task_bar.stim.items()), 2)
        visual_mock.TextStim.assert_called_once()
        visual_mock.Line.assert_called_once()

        phrase = 'Hello, world'
        task_bar.update(phrase)
        self.assertEqual(text_stim_mock.text, phrase)

        task_bar.draw()
        text_stim_mock.draw.assert_called_once()
        border_mock.draw.assert_called_once()

    @patch('bcipy.display.components.task_bar.visual')
    def test_calibration_task_bar(self, visual_mock):
        """Test Calibration Task Bar"""
        text_stim_mock = Mock()
        border_mock = Mock()

        visual_mock.TextStim.return_value = text_stim_mock
        visual_mock.Line.return_value = border_mock

        win = Mock(size=(500, 500), units="norm")
        task_bar = CalibrationTaskBar(win,
                                      inquiry_count=100,
                                      current_index=1,
                                      font='Menlo')

        visual_mock.TextStim.assert_called_once_with(
            alignText='left',
            anchorHoriz='left',
            color='white',
            font='Menlo',
            height=0.1,
            pos=task_bar.layout.left_middle,
            text=' 1/100',
            units='norm',
            win=win)
        visual_mock.Line.assert_called_once()
        self.assertEqual(len(task_bar.stim.items()), 2)
        self.assertEqual(task_bar.stim['task_text'], text_stim_mock)

        task_bar.draw()
        text_stim_mock.draw.assert_called_once()
        border_mock.draw.assert_called_once()

        task_bar.update()
        self.assertEqual(task_bar.displayed_text(), " 2/100")
        self.assertEqual(task_bar.stim['task_text'].text, " 2/100")

    @patch('bcipy.display.components.task_bar.visual')
    def test_copy_phrase_task_bar(self, visual_mock):
        """Test Copy Phrase Task Bar"""
        text_stim_mock = Mock()
        spelled_stim_mock = Mock()
        visual_mock.TextStim.side_effect = [text_stim_mock, spelled_stim_mock]
        visual_mock.Line.return_value = Mock()

        win = Mock(size=(500, 500), units="norm")
        task_bar = CopyPhraseTaskBar(win,
                                     task_text='HELLO_WORLD',
                                     spelled_text='HELLO_')

        self.assertEqual(2, visual_mock.TextStim.call_count)
        visual_mock.Line.assert_called_once()
        self.assertEqual(len(task_bar.stim.items()), 3)

        self.assertEqual(task_bar.displayed_text(), 'HELLO_     ')

        task_bar.update('HELLO_W')
        self.assertEqual(task_bar.spelled_text, 'HELLO_W')
        self.assertEqual(spelled_stim_mock.text, 'HELLO_W    ')


if __name__ == '__main__':
    unittest.main()
