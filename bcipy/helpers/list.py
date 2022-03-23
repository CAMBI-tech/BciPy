"""Utility functions for list processing."""
from typing import Callable, List
from itertools import zip_longest



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


def grouper(iterable, chunk_size, incomplete="fill", fillvalue=None):
    "Collect data into non-overlapping fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, fillvalue='x') --> ABC DEF Gxx
    # grouper('ABCDEFG', 3, incomplete='strict') --> ABC DEF ValueError
    # grouper('ABCDEFG', 3, incomplete='ignore') --> ABC DEF
    chunks = [iter(iterable)] * chunk_size
    if incomplete == "fill":
        return zip_longest(*chunks, fillvalue=fillvalue)
    if incomplete == "strict":
        return zip(*chunks, strict=True)
    if incomplete == "ignore":
        return zip(*chunks)

    raise ValueError("Expected fill, strict, or ignore")
