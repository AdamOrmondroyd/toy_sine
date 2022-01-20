"""
## Recreating the toy sine example from https://arxiv.org/pdf/1506.09024.pdf

Linear interpolation function.
"""

from constants import wavelength
import numpy as np


def f_end_nodes(x, params):
    """
    Vectorized linear interpolation function using n nodes.

    params in format [y0, x1, y1, x2, y2, ..., xn, yn, yn+1] for n internal nodes
    """
    n = len(params) // 2 - 1
    return np.interp(
        x,
        np.concatenate(([0], params[1 : 2 * n + 1 : 2], [wavelength])),
        np.concatenate((params[0 : 2 * n + 2 : 2], params[-1:])),
    )


def get_theta_n(params):
    """
    Extracts the first n parameters from

    params = [n, y0, x1, y1, x2, y2, ..., x_nmax, y_nmax, y_nmax+1]

    where nmax is the maximum value of ceil(n).

    returns theta_n = [y0, x1, y1, x2, y2, ..., x_ceil(n), y_ceil(n), y_nmax+1]
    """
    n = np.ceil(params[0]).astype(int)
    theta_n = np.concatenate(
        (
            params[1 : 2 * n + 2],  # internal x and y
            params[-1:],  # y end node
        )
    )
    return theta_n


def super_model(x, params):
    """
    Super model which allows the number of parameters being used to vary.

    The first element of params is n, the number of interior nodes used in
    the linear interpolation model. This is then used to select the
    appropriate other elements of params to pass to f_end_nodes()

    params = [n, [θ1], [θ2], ..., [θN], y0, y_N+1], since the end points can
    be shared between the models with varying n (I think)
    """
    theta_n = get_theta_n(params)
    return f_end_nodes(x, theta_n)
