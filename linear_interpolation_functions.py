"""
## Recreating the toy sine example from https://arxiv.org/pdf/1506.09024.pdf

Linear interpolation function.
"""

from constants import wavelength
from numpy import concatenate, interp


def f_end_nodes(x, params):
    """
    Vectoried linear interpolstion function using end nodes.

    params = [x1, x2, xn, ..., y0, y1, y2, ..., yn, yn+1]
    """
    n = len(params) // 2 - 1
    return interp(
        x,
        concatenate(([0], params[:n], [wavelength])),
        params[n:],
    )
