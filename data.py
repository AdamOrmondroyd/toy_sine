"""
## Recreating the toy sine example from https://arxiv.org/pdf/1506.09024.pdf

Get data from sine_data.npy or line_data.npy.
"""

from numpy import load
from constants import line_or_sine

filename = "data_{}.npy".format(line_or_sine)
data = load(filename)
xs, ys = data[0], data[1]
