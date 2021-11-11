"""
## Recreating the toy sine example from https://arxiv.org/pdf/1506.09024.pdf

Get data from sine_data.npy or line_data.npy.
"""
import os
from pathlib import Path
from numpy import load


def get_data(line_or_sine, x_errors):
    """Returns tuple xs, ys for either "line" or "sine"."""
    if x_errors:
        filename = Path(__file__).parent.joinpath("data_{}_x_errors.npy".format(line_or_sine))
    else:
        filename = Path(__file__).parent.joinpath("data_{}.npy".format(line_or_sine))
    data = load(filename)
    return data[0], data[1]
