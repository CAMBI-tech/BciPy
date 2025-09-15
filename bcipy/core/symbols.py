"""Defines helper methods and variables related to input symbols"""
import os
from string import ascii_uppercase
from typing import Any, Callable, List, Optional

SPACE_CHAR = '_'
BACKSPACE_CHAR = '<'


def alphabet(parameters: Optional[Any] = None, include_path: bool = True,
             backspace: str = BACKSPACE_CHAR, space: str = SPACE_CHAR) -> List[str]:
    """Standardizes and returns the alphabet symbols used in BciPy.

    The symbols can either be text (uppercase ASCII letters, backspace, and space)
    or paths to image files, depending on the `parameters` and `is_txt_stim` setting.

    Args:
        parameters (Optional[Any], optional): A dictionary-like object containing configuration
            parameters, specifically 'is_txt_stim' and 'path_to_presentation_images'.
            Defaults to None.
        include_path (bool, optional): If True and image stimuli are used, returns full paths to images.
            If False, returns just the image filenames without extensions. Defaults to True.
        backspace (str, optional): The character representing backspace. Defaults to `BACKSPACE_CHAR`.
        space (str, optional): The character representing space. Defaults to `SPACE_CHAR`.

    Returns:
        List[str]: A list of alphabet symbols (either letters or image paths).
    """
    if parameters and not parameters['is_txt_stim']:
        # construct an array of paths to images
        path = parameters['path_to_presentation_images']
        stimulus_array: List[str] = []
        for stimulus_filename in sorted(os.listdir(path)):
            # PLUS.png is reserved for the fixation symbol
            root, ext = os.path.splitext(stimulus_filename)
            if ext in [".png", ".bmp", ".jpg"] and not root.endswith("PLUS"):
                if include_path:
                    img = os.path.join(path, stimulus_filename)
                else:
                    img = root
                stimulus_array.append(img)
        return stimulus_array

    return list(ascii_uppercase) + [backspace, space]


def qwerty_order(is_txt_stim: bool = True,
                 space: str = SPACE_CHAR,
                 backspace: str = BACKSPACE_CHAR) -> Callable[[str], int]:
    """Returns a function that can be used to sort alphabet symbols in QWERTY order.

    Note that sorting only works for text stimuli.

    Args:
        is_txt_stim (bool, optional): If True, indicates text stimuli. Defaults to True.
        space (str, optional): The character representing space. Defaults to `SPACE_CHAR`.
        backspace (str, optional): The character representing backspace. Defaults to `BACKSPACE_CHAR`.

    Returns:
        Callable[[str], int]: A function that takes a symbol string and returns its QWERTY index.

    Raises:
        NotImplementedError: If `is_txt_stim` is False, as QWERTY ordering is not implemented for images.
    """
    if not is_txt_stim:
        raise NotImplementedError('QWERTY ordering not implemented for images')

    row1 = "QWERTYUIOP"
    row2 = f"ASDFGHJKL{backspace}"
    row3 = f" ZXCV{space}BNM "
    return f"{row1}{row2}{row3}".index


def frequency_order(
        is_txt_stim: bool = True,
        space: str = SPACE_CHAR,
        backspace: str = BACKSPACE_CHAR) -> Callable[[str], int]:
    """Returns a function that can be used to sort alphabet symbols by frequency of use in English."

    Note that sorting only works for text stimuli.

    Args:
        is_txt_stim (bool, optional): If True, indicates text stimuli. Defaults to True.
        space (str, optional): The character representing space. Defaults to `SPACE_CHAR`.
        backspace (str, optional): The character representing backspace. Defaults to `BACKSPACE_CHAR`.

    Returns:
        Callable[[str], int]: A function that takes a symbol string and returns its frequency order index.

    Raises:
        NotImplementedError: If `is_txt_stim` is False, as frequency ordering is not implemented for images.
    """
    if not is_txt_stim:
        raise NotImplementedError(
            'Frequency ordering not implemented for images')
    return f"ETAOINSHRDLCUMWFGYPBVKJXQZ{backspace}{space}".index


DEFAULT_SYMBOL_SET = alphabet()
