"""Utility functions for list processing."""
from itertools import zip_longest
from typing import Any, Callable, List, Optional, Union


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
    # grouper('ABCDEFG', 3, incomplete='ignore') --> ABC DEF
    chunks = [iter(iterable)] * chunk_size
    if incomplete == "fill":
        if fillvalue:
            return zip_longest(*chunks, fillvalue=fillvalue)
        raise ValueError('fillvalue must be defined if using incomplete=fill')
    if incomplete == "ignore":
        return zip(*chunks)

    raise ValueError("Expected fill or ignore")


def find_index(iterable: List,
               match_item: Union[Any, Callable],
               key: Callable = lambda x: x) -> Optional[int]:
    """Find the index of the first item in the iterable which matches."""
    for i, value in enumerate(iterable):
        if callable(match_item):
            result = match_item(value)
        else:
            result = key(value) == match_item
        if result:
            return i
    return None


def swapped(lst: List[Any], index1: int, index2: int) -> List[Any]:
    """Creates a copy of the provided list with elements at the given indices
    swapped."""
    replacements = {index1: lst[index2], index2: lst[index1]}
    return [replacements.get(i, val) for i, val in enumerate(lst)]


def expanded(lst: List[Any],
             length: int,
             fill: Union[Any, Callable] = lambda x: x[-1]) -> List[Any]:
    """Creates a copy of the provided list expanded to the given length. By
    default the last item is used as the fill item.

    Parameters
    ----------
        lst - list of items to copy
        length - expands list to this length
        fill - optional; used to determine which element to use for
            the fill, given the list. Defaults to the last element.

    >>> expand([1,2,3], length=5)
    [1,2,3,3,3]
    """
    times = length - len(lst)
    if lst and times > 0:
        item = fill(lst) if callable(fill) else fill
        return lst + ([item] * times)
    return lst
