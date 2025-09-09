import unittest

from bcipy.core.symbols import alphabet
from typing import Dict

class TestAlphabet(unittest.TestCase):
    def test_alphabet_text(self):
        parameters = {}

        parameters['is_txt_stim'] = True

        alp = alphabet(parameters)

        self.assertEqual(alp, [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '<', '_'
        ])

    def test_alphabet_images_with_path(self):
        path = 'bcipy/core/tests/resources/images/'
        parameters: Dict[str, bool | str] = {}
        parameters['is_txt_stim'] = False
        parameters['path_to_presentation_images'] = path

        alp = alphabet(parameters)

        self.assertEqual(alp, [
            path + 'a_1x1.bmp',
            path + 'b_1x1.jpg',
            path + 'c_1x1.png'
        ])

    def test_alphabet_images_without_path(self):
        path = 'bcipy/core/tests/resources/images/'
        parameters: Dict[str, bool | str] = {}
        parameters['is_txt_stim'] = False
        parameters['path_to_presentation_images'] = path

        alp = alphabet(parameters, include_path=False)

        self.assertEqual(alp, ['a_1x1', 'b_1x1', 'c_1x1'])
