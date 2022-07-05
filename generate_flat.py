"""
## Recreating the toy sine example from https://arxiv.org/pdf/1506.09024.pdf

it occurred to me that a flat case would also be good, to make sure it comgerges to N=0,
as I have just changed linf to (hopefully) properly deal with N=0
"""

import numpy as np
from numpy.random import default_rng
from constants import amplitude, n_points, sigma_x, sigma_y, wavelength


def noisy_flat(
    n_points,
    x_errors,
    wavelength=wavelength,
    sigma_x=sigma_x,
    sigma_y=sigma_y,
):
    """Returns tuple xs, ys where xs are uniformally distributed over a wavelength,
    and ys are just Gaussian noise"""
    xs = default_rng().uniform(0.0, wavelength, n_points)
    xs += default_rng().normal(0, sigma_x * x_errors, n_points)
    ys = default_rng().normal(0, sigma_y, n_points)
    ys = ys[(xs >= 0) & (xs <= wavelength)]
    xs = xs[(xs >= 0) & (xs <= wavelength)]
    return xs, ys


if __name__ == "__main__":
    for x_errors in False, True:
        filename = f"data_flat{'_x_errors' if x_errors else ''}.npy"
        xs, ys = noisy_flat(n_points, x_errors, wavelength, sigma_x, sigma_y)
        np.save(filename, np.stack((xs, ys)))
