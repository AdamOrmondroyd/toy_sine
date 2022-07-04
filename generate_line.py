"""
## Recreating the toy sine example from https://arxiv.org/pdf/1506.09024.pdf

Generates data_line.npy.
"""

import numpy as np
from numpy.random import default_rng
from constants import amplitude, n_points, sigma_x, sigma_y, wavelength
from linf import Linf


def noisy_line(
    n_points,
    x_errors,
    amplitude=amplitude,
    wavelength=wavelength,
    sigma_x=sigma_x,
    sigma_y=sigma_y,
):
    """Analogous to noisy_sine but using the "line" function from https://arxiv.org/pdf/1506.09024.pdf ."""
    xs = default_rng().uniform(0.0, wavelength, n_points)
    ys = Linf(0, wavelength)(
        xs, [0, wavelength / 4, amplitude, wavelength * 3 / 4, -amplitude, 0]
    )
    xs += default_rng().normal(0, sigma_x * x_errors, n_points)
    ys += default_rng().normal(0, sigma_y, n_points)
    # xs = xs % wavelength
    ys = ys[(xs >= 0) & (xs <= wavelength)]
    xs = xs[(xs >= 0) & (xs <= wavelength)]
    return xs, ys


if __name__ == "__main__":
    for x_errors in False, True:
        if x_errors:
            filename = "data_line_x_errors.npy"
        else:
            filename = "data_line.npy"
        xs, ys = noisy_line(n_points, x_errors, amplitude, wavelength, sigma_x, sigma_y)
        np.save(filename, np.stack((xs, ys)))
