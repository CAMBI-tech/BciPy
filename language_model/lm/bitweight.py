#!/usr/bin/env python
# bitweight.py: class for -log_2 math
# Kyle Gorman <gormanky@ohsu.edu> and Steven Bedrick <bedricks@ohsu.edu>

from __future__ import division

from math import exp, log


INF = float("inf")


def log2(x):
    return log(x, 2)


def exp2(x):
    return 2 ** x


class BitWeight(object):

    """
    A real-space value represented internally as a negative base-2 
    logarithm, similar to that used in the log probability semiring,
    and used to avoid underflow. Note that this can only store 
    non-negative reals; log 0 == -inf, and logs of negative numbers are 
    undefined. For the same reason, __neg__ (one-place '-' operator), 
    __sub__ (two-place '-' operator), and __isub__ (two-place '-' with 
    assignment operator) have all been left unimplemented.
    """

    BIG = log2(1e63 - 1)  # largest C-like integer on 64-bit platforms

    def __init__(self, val=0., is_neg_log=False):
        """
        If called without any arguments, the value is 0. in real number
        space. 

        If is_neg_log is True, just store val because we've been given
        a number that's already been converted.

        >>> BitWeight(.5)
        BitWeight(bw=1.0)
        >>> BitWeight(-.5)
        Traceback (most recent call last):
        ...
        ValueError: Cannot real-ify "-0.5"
        >>> BitWeight(0.).to_real == 0
        True
        """
        if is_neg_log:
            self.bw = val
        else:
            try:
                # check for non-negative real arguments
                if val < 0.:
                    raise ValueError
                elif val == 0.:
                    self.bw = INF
                else:
                    self.bw = -log2(val)
            except (ValueError, TypeError):
                raise ValueError('Cannot real-ify "{}"'.format(val))

    def __repr__(self):
        return '{}(bw={})'.format(self.__class__.__name__, self.bw)

    @property
    def to_real(self):
        return exp2(-self.bw)

    def __add__(self, other):
        """
        Addition in real space; an optimization of Manning & Schuetze,
        p. 337 (eq. 9.21)
        
        >>> a_real = .5
        >>> b_real = .25
        >>> a_bw = BitWeight(a_real)
        >>> b_bw = BitWeight(b_real)
        >>> BitWeight.close_enough((a_bw + b_bw).to_real, a_real + b_real)
        True
        >>> (BitWeight(.25) + BitWeight(.25)).to_real
        0.5
        """
        other_bw = other if hasattr(other, 'bw') else BitWeight(other)
        if other_bw.bw - self.bw > self.BIG:
            to_return = self.bw
        elif self.bw - other_bw.bw > self.BIG:
            to_return = other_bw.bw
        else:
            if other_bw.bw > self.bw:
                to_return = other_bw.bw - log2(1. + \
                            exp2(other_bw.bw - self.bw))
            elif other_bw.bw < self.bw:
                to_return = self.bw - log2(exp2(self.bw - other_bw.bw) + 1.)
            else:
                to_return = other_bw.bw - 1.
                # not 1 + x_bw.bw as you might think, as BWs are
                # NEGATIVE log-weights
        return BitWeight(to_return, True)

    def __radd__(self, other):
        """
        Reflected (other + self) real-space addition
        """
        return self + other 

    def __iadd__(self, other):
        """
        In-place (+=) real-space addition    

        >>> a_real = .5
        >>> b_real = .25
        >>> a_bw = BitWeight(a_real)
        >>> a_bw += b_real
        >>> BitWeight.close_enough(a_bw.to_real, a_real + b_real)
        True
        """
        return self + other 

    def __mul__(self, other):
        """
        Multiplication in real space via log-addition
        
        >>> a_real = .5
        >>> b_real = .25
        >>> a_bw = BitWeight(a_real)
        >>> b_bw = BitWeight(b_real)
        >>> BitWeight.close_enough((a_bw * b_bw).to_real, a_real * b_real)
        True
        """
        other_bw = other if hasattr(other, 'bw') else BitWeight(other)
        return BitWeight(self.bw + other_bw.bw, True)

    def __rmul__(self, other):
        """
        Reflected (other * self) real-space multiplication
        """
        return self * other

    def __imul__(self, other):
        """
        In-place (*=) real-space multiplication

        >>> a_real = .5
        >>> b_real = .25
        >>> a_bw = BitWeight(a_real)
        >>> a_bw *= b_real
        >>> BitWeight.close_enough(a_bw.to_real, a_real * b_real)
        True
        """
        return self * other 

    def __truediv__(self, other):
        """
        Division (in real space)

        Probability space values:

        >>> a_real = .5
        >>> b_real = .25

        Small integer values (including 1, which has special behaviors):

        >>> c_real = 1
        >>> d_real = 4

        Large integer values:

        >>> e_real = 100
        >>> f_real = 400
        
        Convert to BitWeight:

        >>> a_bw = BitWeight(a_real)
        >>> b_bw = BitWeight(b_real)
        >>> c_bw = BitWeight(c_real)
        >>> d_bw = BitWeight(d_real)
        >>> e_bw = BitWeight(e_real)
        >>> f_bw = BitWeight(f_real)
                
        Tests:

        >>> BitWeight.close_enough((a_bw / b_bw).to_real, a_real / b_real)
        True
        >>> BitWeight.close_enough((b_bw / a_bw).to_real, b_real / a_real)
        True
        >>> BitWeight.close_enough((c_bw / d_bw).to_real, c_real / d_real)
        True
        >>> BitWeight.close_enough((d_bw / c_bw).to_real, d_real / c_real)
        True
        >>> BitWeight.close_enough((e_bw / f_bw).to_real, e_real / f_real)
        True
        >>> BitWeight.close_enough((f_bw / e_bw).to_real, f_real / e_real)
        True
        """
        other_bw = other if hasattr(other, 'bw') else BitWeight(other)
        return BitWeight(self.bw - other_bw.bw, True)

    def __itruediv__(self, other):
        """
        In-place (/=) real-space division

        >>> a_real = 1
        >>> b_real = 4
        >>> a_bw = BitWeight(a_real)
        >>> b_bw = BitWeight(b_real)
        >>> a_bw /= b_bw
        >>> BitWeight.close_enough(a_bw.to_real, a_real / b_real)
        True
        """
        return self / other 

    def __pow__(self, other):
        """
        Real-Space Exponentiation

        >>> bw = BitWeight(0.5)
        >>> e_bw = bw ** 3
        >>> BitWeight.close_enough(e_bw.to_real, 0.125)
        True
        """
        return BitWeight(self.bw * other, True)

    @staticmethod
    def close_enough(x, y, tol=1e-8):
        """
        Simple test for float approximate-equality, using _relative 
        difference_, defined as the absolute difference of the two terms 
        normalized by the average of the two terms. See:

        http://en.wikipedia.org/wiki/Relative_difference
        """
        return 2. * abs(x - y) / (abs(x) + abs(y)) < tol

    def __cmp__(self, other):
        """
        >>> a_real = .5
        >>> b_real = .0001
        >>> a_bw = BitWeight(a_real)
        >>> b_bw = BitWeight(b_real)
        >>> a_bw > b_bw
        True
        >>> b_bw == a_bw
        False
        >>> BitWeight(a_real - b_real) < b_bw
        False
        >>> a_bw > BitWeight(a_real - b_real)
        True
        >>> a_bw >= b_bw
        True
        >>> b_bw <= a_bw
        True
        >>> a_bw >= BitWeight(0.5)
        True
        >>> a_bw = BitWeight(1)
        >>> b_bw = BitWeight(1)
        >>> cmp(a_bw, b_bw)
        0
        """
        other_bw = other if hasattr(other, 'bw') else BitWeight(other)
        # avoid stray NaN caused by INF - INF
        if other_bw.bw == INF and self.bw == INF:
            delta = 0
        else:
            delta = other_bw.bw - self.bw
        delta = other_bw.bw - self.bw
        if delta == 0:
            return 0
        elif delta < 0:
            return -1
        else: # delta > 0
            return 1

    # "Rich comparisons" for Python 3 support
    # TODO: should use close_enough() here?
    def __eq__(self, other):
        other_bw = other if hasattr(other, 'bw') else BitWeight(other)
        return other_bw.bw == self.bw 
        
    def __ne__(self, other):
        return not (self == other)

    # remember: in negative log space, bigger is actually smaller, e.g.,
    # BitWeight(0.5).bw -> 1.0
    # BitWeight(0.25).bw -> 2.0
    def __lt__(self, other):
        other_bw = other if hasattr(other, 'bw') else BitWeight(other)
        return self.bw > other_bw.bw
                    
    def __le__(self, other):
        return (self == other) or (self < other)
            
    def __gt__(self, other):
        other_bw = other if hasattr(other, 'bw') else BitWeight(other)
        return self.bw < other_bw.bw
            
    def __ge__(self, other):
        return (self == other) or (self > other)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
