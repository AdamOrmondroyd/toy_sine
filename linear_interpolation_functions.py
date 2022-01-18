"""
## Recreating the toy sine example from https://arxiv.org/pdf/1506.09024.pdf

Linear interpolation function.
"""

from constants import wavelength
from numpy import ceil, concatenate, interp


def f_end_nodes(x, params):
    """
    Vectorised linear interpolstion function using end nodes.

    params = [x1, x2, xn, ..., y0, y1, y2, ..., yn, yn+1]
    """
    n = len(params) // 2 - 1
    return interp(
        x,
        concatenate(([0], params[:n], [wavelength])),
        params[n:],
    )


def get_theta_n(params):
    n = ceil(params[0]).astype(int)
    theta = params[1:]
    start = 0
    middle = n
    end = 2 * n
    theta_n = concatenate(
        (
            theta[start:middle],  # internal x
            theta[-2:-1],  # y0
            theta[middle:end],  # internal y
            theta[-1:],  # yn+1
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
