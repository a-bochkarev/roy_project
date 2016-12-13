from __future__ import absolute_import, print_function, division

from .solver import Solver


class SimpleFill(Solver):
    def __init__(self, fill_method="mean", min_value=None, max_value=None):
        """
        Possible values for fill_method:
            "zero": fill missing entries with zeros
            "mean": fill with column means
            "median" : fill with column medians
            "min": fill with min value per column
            "random": fill with gaussian noise according to mean/std of column
        """
        Solver.__init__(
            self,
            fill_method=fill_method,
            min_value=None,
            max_value=None)

    def solve(self, X, missing_mask):
        """
        Since X is given to us already filled, just return it.
        """
        return X
