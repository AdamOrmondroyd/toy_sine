"""
## Recreating the toy sine example from https://arxiv.org/pdf/1506.09024.pdf

Generates data_sine.npy.
"""

from numpy import save, sin, stack, pi
from numpy.random import default_rng
from constants import amplitude, n_points, sigma_x, sigma_y, wavelength

filename = "data_sine.npy"


def noisy_sine(
    n_points,
    x_errors,
    amplitude=amplitude,
    wavelength=wavelength,
    sigma_x=sigma_x,
    sigma_y=sigma_y,
):
    """Returns tuple xs, ys where xs are uniformally distributed over a wavelength,
    and ys are sin(xs) + Gaussian noise"""
    xs = default_rng().uniform(0.0, wavelength, n_points)
    ys = amplitude * sin(2 * pi / wavelength * xs)
    xs += default_rng().normal(0, sigma_x * x_errors, n_points)
    ys += default_rng().normal(0, sigma_y, n_points)
    xs = xs % wavelength
    return xs, ys


if __name__ == "__main__":
    for x_errors in False, True:
        if x_errors:
            filename = "data_sine_x_errors.npy"
        else:
            filename = "data_sine.npy"
        xs, ys = noisy_sine(n_points, x_errors, amplitude, wavelength, sigma_x, sigma_y)
        save(filename, stack((xs, ys)))
