"""Defines helper methods and variables related to input symbols"""
import os
from string import ascii_uppercase
from typing import Callable

SPACE_CHAR = '_'
BACKSPACE_CHAR = '<'


def alphabet(parameters=None, include_path=True):
    """Alphabet.

    Function used to standardize the symbols we use as alphabet.

    Returns
    -------
        array of letters.
    """
    if parameters and not parameters['is_txt_stim']:
        # construct an array of paths to images
        path = parameters['path_to_presentation_images']
        stimulus_array = []
        for stimulus_filename in sorted(os.listdir(path)):
            # PLUS.png is reserved for the fixation symbol
            if stimulus_filename.endswith(
                    '.png') and not stimulus_filename.endswith('PLUS.png'):
                if include_path:
                    img = os.path.join(path, stimulus_filename)
                else:
                    img = os.path.splitext(stimulus_filename)[0]
                stimulus_array.append(img)
        return stimulus_array

    return list(ascii_uppercase) + [BACKSPACE_CHAR, SPACE_CHAR]


def qwerty_order(is_txt_stim: bool = True,
                 space: str = SPACE_CHAR,
                 backspace: str = BACKSPACE_CHAR) -> Callable:
    """Returns a function that can be used to sort the alphabet symbols
    in QWERTY order. Note that sorting only works for text stim.
    """
    if not is_txt_stim:
        raise NotImplementedError('QWERTY ordering not implemented for images')

    row1 = "QWERTYUIOP"
    row2 = f"ASDFGHJKL{backspace}"
    row3 = f" ZXCV{space}BNM "
    return f"{row1}{row2}{row3}".index


def frequency_order(is_txt_stim: bool = True) -> Callable:
    """Returns a function that can be used to sort the alphabet symbols
    in most frequently used order in the English language.
    """
    if not is_txt_stim:
        raise NotImplementedError(
            'Frequency ordering not implemented for images')
    return f"ETAOINSHRDLCUMWFGYPBVKJXQZ{BACKSPACE_CHAR}{SPACE_CHAR}".index


DEFAULT_SYMBOL_SET = alphabet()
