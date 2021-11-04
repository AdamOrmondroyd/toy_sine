"""
## Recreating the toy sine example from https://arxiv.org/pdf/1506.09024.pdf

Generates data_line.npy.
"""

from numpy import load, save, stack, pi
from numpy.random import default_rng
from constants import amplitude, n_points, sigma_x, sigma_y, wavelength
from linear_interpolation_functions import f_interp as f

filename = "data_line.npy"


def noisy_line(
    n_points,
    amplitude=amplitude,
    wavelength=wavelength,
    sigma_x=sigma_x,
    sigma_y=sigma_y,
):
    """Analogous to noisy_sine but using the "line" function from https://arxiv.org/pdf/1506.09024.pdf ."""
    xs = default_rng().uniform(0.0, wavelength, n_points)
    ys = f(xs, [wavelength / 4, wavelength * 3 / 4, amplitude, -amplitude])
    xs += default_rng().normal(0, sigma_x, n_points)
    ys += default_rng().normal(0, sigma_y, n_points)
    xs = xs % wavelength
    return xs, ys


if __name__ == "__main__":
    xs, ys = noisy_line(n_points, amplitude, wavelength, sigma_x, sigma_y)
    save(filename, stack((xs, ys)))
