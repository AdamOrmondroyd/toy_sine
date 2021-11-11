"""
## Recreating the toy sine example from https://arxiv.org/pdf/1506.09024.pdf

Linear interpolation function.
"""

from constants import wavelength
from numpy import (
    concatenate,
    interp,
    greater,
    logical_and,
    less_equal,
    outer,
)


def f_end_nodes_numpy(x, params):
    n = len(params) // 2 - 1
    return interp(
        x,
        concatenate(([0], params[:n], [wavelength])),
        params[n:],
    )


def f_cyclic_numpy(x, params):
    """Vectorised cyclic linear interpolation function using np.interp."""
    n = len(params) // 2
    return interp(
        x,
        params[:n],
        params[n:],
        period=wavelength,
    )


def f_cyclic_adam(x, params):
    """Vectorised cyclic linear interpolation function done manually."""
    x %= wavelength
    n = len(params) // 2
    xp = params[:n]
    yp = params[n:]
    xp = concatenate((xp[-1:] - wavelength, xp, xp[0:1] + wavelength))
    yp = concatenate((yp[-1:], yp, yp[0:1]))
    ms = (yp[1:] - yp[:-1]) / (xp[1:] - xp[:-1])
    cs = yp[:-1] - ms * xp[:-1]

    return (outer(x, ms) + cs)[
        logical_and(less_equal.outer(xp[:-1], x), greater.outer(xp[1:], x)).T
    ]


if __name__ == "__main__":
    from numpy import array
    from data import xs

    params = array([wavelength / 4, 3 * wavelength / 4, 1.0, -1.0])
    print(f_cyclic_adam(xs, params))
    exit()
