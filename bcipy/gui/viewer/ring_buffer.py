"""Defines a RingBuffer with a fixed size; when full additional elements
overwrite the oldest items in the data structure.

Adapted from Python Cookbook by David Ascher, Alex Martelli
https://www.oreilly.com/library/view/python-cookbook/0596001673/ch05s19.html
"""


class RingBuffer:
    """Data structure with a fixed size; when full additional elements
    overwrite the oldest items in the data structure.
    
    Parameters
    ----------
        size_max - max size of the buffer
        pre_allocated - whether to create all values on initialization
        empty_value - if pre_allocated, empty_value is used to set the
            values with no data.
    """

    def __init__(self, size_max: int, pre_allocated: bool = False, empty_value=None):
        assert size_max > 0
        self.empty_value = empty_value
        self.max = size_max
        self.data = [empty_value] * size_max if pre_allocated else []
        self.cur = 0
        self.full = False
        self.pre_allocated = pre_allocated

    def append(self, item):
        """Add an element to the buffer, overwriting if full."""
        if self.full or self.pre_allocated:
            # overwrite
            self.data[self.cur] = item
        else:
            self.data.append(item)
        if not self.full:
            self.full = self.cur == self.max-1
        self.cur = (self.cur+1) % self.max

    def get(self):
        """Return a list of elements from the oldest to the newest."""
        if self.full:
            return self.data[self.cur:]+self.data[:self.cur]
        return self.data

    def is_empty(self):
        return len(self.data) == 0 or self.data[0] == self.empty_value