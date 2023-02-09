import unittest

from bcipy.language.main import alphabet

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

    def test_alphabet_images(self):
        parameters = {}
        parameters['is_txt_stim'] = False
        parameters['path_to_presentation_images'] = ('bcipy/static/images/'
                                                     'rsvp/')

        alp = alphabet(parameters)

        self.assertNotEqual(alp, [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', '<', '_'
        ])