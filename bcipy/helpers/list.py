"""Utility functions for list processing."""
from typing import Callable, List


def destutter(items: List, key: Callable = lambda x: x) -> List:
    """Removes sequential duplicates from a list. Retains the last item in the
    sequence. Equality is determined using the provided key function.

    Parameters
    ----------
        items - list of items with sequential duplicates
        key - equality function
    """
    deduped = []
    for item in items:
        if len(deduped) == 0 or key(item) != key(deduped[-1]):
            deduped.append(item)
        else:
            deduped[-1] = item
    return deduped
