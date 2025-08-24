"""Utility functions for list processing."""
from itertools import zip_longest
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union


def destutter(items: List[Any], key: Callable = lambda x: x) -> List:
    """Removes sequential duplicates from a list. Retains the last item in the
    sequence.

    Equality is determined using the provided key function.

    Args:
        items (List[Any]): List of items with sequential duplicates.
        key (Callable, optional): Equality function. Defaults to `lambda x: x`.

    Returns:
        List: A new list with sequential duplicates removed.
    """
    deduped: List[Any] = []
    for item in items:
        if len(deduped) == 0 or key(item) != key(deduped[-1]):
            deduped.append(item)
        else:
            deduped[-1] = item
    return deduped


def grouper(iterable: Any, chunk_size: int, incomplete: str = "fill",
            fillvalue: Optional[Any] = None) -> Union[Iterator[Tuple], Iterator[Any]]:
    """Collect data into non-overlapping fixed-length chunks or blocks.

    Args:
        iterable (Any): The iterable to group.
        chunk_size (int): The size of each chunk.
        incomplete (str, optional): Strategy for incomplete chunks. Can be "fill" or "ignore".
            Defaults to "fill".
        fillvalue (Optional[Any], optional): Value to fill incomplete chunks with if `incomplete` is "fill".
            Defaults to None.

    Returns:
        Union[Iterator[Tuple], Iterator[Any]]: An iterator yielding chunks.

    Raises:
        ValueError: If `fillvalue` is not defined when `incomplete` is "fill", or if `incomplete`
                    is neither "fill" nor "ignore".

    Examples:
        grouper('ABCDEFG', 3, fillvalue='x') --> ABC DEF Gxx
        grouper('ABCDEFG', 3, incomplete='ignore') --> ABC DEF
    """
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
    """Find the index of the first item in the iterable which matches.

    Args:
        iterable (List): The list to search through.
        match_item (Union[Any, Callable]): The item to match or a callable to apply to each item.
        key (Callable, optional): A function to apply to each item before comparison. Defaults to `lambda x: x`.

    Returns:
        Optional[int]: The index of the first matching item, or None if no match is found.
    """
    for i, value in enumerate(iterable):
        if callable(match_item):
            result = match_item(value)
        else:
            result = key(value) == match_item
        if result:
            return i
    return None


def swapped(lst: List[Any], index1: int, index2: int) -> List[Any]:
    """Creates a copy of the provided list with elements at the given indices swapped.

    Args:
        lst (List[Any]): The original list.
        index1 (int): The index of the first element to swap.
        index2 (int): The index of the second element to swap.

    Returns:
        List[Any]: A new list with the elements at `index1` and `index2` swapped.
    """
    replacements = {index1: lst[index2], index2: lst[index1]}
    return [replacements.get(i, val) for i, val in enumerate(lst)]


def expanded(lst: List[Any],
             length: int,
             fill: Union[Any, Callable] = lambda x: x[-1]) -> List[Any]:
    """Creates a copy of the provided list expanded to the given length.

    By default, the last item is used as the fill item.

    Args:
        lst (List[Any]): List of items to copy.
        length (int): The target length to expand the list to.
        fill (Union[Any, Callable], optional): Used to determine which element to use for
            the fill, given the list. Defaults to the last element.

    Returns:
        List[Any]: The expanded list.

    Examples:
        >>> expanded([1,2,3], length=5)
        [1,2,3,3,3]
    """
    times = length - len(lst)
    if lst and times > 0:
        item = fill(lst) if callable(fill) else fill
        return lst + ([item] * times)
    return lst


def pairwise(iterable: Any) -> Iterator[Tuple]:
    """Returns an iterator over overlapping pairs from the input iterable.

    Args:
        iterable (Any): The iterable to process.

    Yields:
        Tuple: A tuple containing two consecutive elements from the iterable.

    Examples:
        pairwise('ABCDEFG') → AB BC CD DE EF FG
        https://docs.python.org/3/library/itertools.html#itertools.pairwise
    """
    iterator = iter(iterable)
    a = next(iterator, None)
    for b in iterator:
        yield a, b
        a = b
