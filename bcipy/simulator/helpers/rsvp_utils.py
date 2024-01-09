from bcipy.helpers.symbols import BACKSPACE_CHAR


def next_target_letter(current_sentence, target_sentence):
    """Computes the next target letter based on the currently spelled_text.
    """
    if target_sentence[0:len(current_sentence)] == current_sentence:
        # if correctly spelled so far, get the next letter.
        return target_sentence[len(current_sentence)] if len(
            current_sentence) < len(target_sentence) else target_sentence[-1]
    return BACKSPACE_CHAR
