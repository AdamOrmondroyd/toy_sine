"""
## Recreating the toy sine example from https://arxiv.org/pdf/1506.09024.pdf

Get data from sine_data.npy or line_data.npy.
"""
import os
from pathlib import Path
from numpy import load


def get_data(line_or_sine):
    """Returns tuple xs, ys for either "line" or "sine"."""
    filename = Path(__file__).parent.joinpath("data_{}.npy".format(line_or_sine))
    data = load(filename)
    return data[0], data[1]
