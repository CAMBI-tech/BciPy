"""Ring buffer implementation for efficient data storage and retrieval.

This module defines a RingBuffer class that implements a fixed-size circular buffer.
When the buffer is full, new elements overwrite the oldest items in the data structure.

Adapted from Python Cookbook by David Ascher, Alex Martelli
https://www.oreilly.com/library/view/python-cookbook/0596001673/ch05s19.html
"""

from typing import Any, List, Optional, TypeVar

T = TypeVar('T')


class RingBuffer:
    """A fixed-size circular buffer implementation.

    This class implements a ring buffer (circular buffer) with a fixed maximum size.
    When the buffer is full, new elements overwrite the oldest items in the data structure.

    Attributes:
        empty_value (Any): Value used to represent empty slots in the buffer.
        max (int): Maximum size of the buffer.
        data (List[Any]): Internal storage for buffer elements.
        cur (int): Current position in the buffer.
        full (bool): Whether the buffer is full.
        pre_allocated (bool): Whether the buffer was pre-allocated with empty values.

    Args:
        size_max (int): Maximum size of the buffer.
        pre_allocated (bool, optional): Whether to create all values on initialization.
            Defaults to False.
        empty_value (Any, optional): If pre_allocated, this value is used to set the
            values with no data. Defaults to None.

    Raises:
        AssertionError: If size_max is not greater than 0.
    """

    def __init__(self, size_max: int, pre_allocated: bool = False,
                 empty_value: Optional[Any] = None) -> None:
        """Initialize the ring buffer.

        Args:
            size_max (int): Maximum size of the buffer.
            pre_allocated (bool, optional): Whether to create all values on initialization.
                Defaults to False.
            empty_value (Any, optional): If pre_allocated, this value is used to set the
                values with no data. Defaults to None.

        Raises:
            AssertionError: If size_max is not greater than 0.
        """
        assert size_max > 0
        self.empty_value = empty_value
        self.max = size_max
        self.data: List[Any] = [empty_value] * \
            size_max if pre_allocated else []
        self.cur = 0
        self.full = False
        self.pre_allocated = pre_allocated

    def append(self, item: Any) -> None:
        """Add an element to the buffer, overwriting if full.

        Args:
            item (Any): The item to add to the buffer.
        """
        if self.full or self.pre_allocated:
            # overwrite
            self.data[self.cur] = item
        else:
            self.data.append(item)
        if not self.full:
            self.full = self.cur == self.max - 1
        self.cur = (self.cur + 1) % self.max

    def get(self) -> List[Any]:
        """Return a list of elements from the oldest to the newest.

        Returns:
            List[Any]: List of elements in chronological order.
        """
        if self.full:
            return self.data[self.cur:] + self.data[:self.cur]
        return self.data

    def is_empty(self) -> bool:
        """Check if the buffer is empty.

        Returns:
            bool: True if the buffer is empty or contains only empty values.
        """
        return len(self.data) == 0 or self.data[0] == self.empty_value
