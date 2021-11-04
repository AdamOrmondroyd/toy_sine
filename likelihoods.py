"""
## Recreating the toy sine example from https://arxiv.org/pdf/1506.09024.pdf

Likelihood functions for fitting to a linear interpolation function with 
y errors or both x and y errors.
"""

from numpy import append, exp, log, outer, pi, subtract, sum, sqrt
from scipy.special import erf
from constants import sigma_x, sigma_y, wavelength
from data import xs, ys
from linear_interpolation_functions import f_numpy as f


LOG_2_SQRT_2PIλ = log(2) + 0.5 * log(2 * pi * wavelength)
var_x, var_y = sigma_x ** 2, sigma_y ** 2


def y_errors_likelihood(xs, ys, params):
    logL = -len(xs) * 0.5 * log(2 * pi * var_y)
    logL += sum(-((ys - f(xs, params)) ** 2) / 2 / var_y)
    return logL, []


def x_y_errors_likelihood(params):
    n = len(params) // 2
    x_nodes = params[:n]
    y_nodes = params[n:]
    y_0 = (
        y_nodes[0]
        - (y_nodes[0] - y_nodes[-1])
        / (x_nodes[0] - (x_nodes[-1] - wavelength))
        * x_nodes[0]
    )

    x_nodes = append(0, append(x_nodes, wavelength))
    y_nodes = append(y_0, append(y_nodes, y_0))
    ms = (y_nodes[1:] - y_nodes[:-1]) / (x_nodes[1:] - x_nodes[:-1])
    cs = y_nodes[:-1] - ms * x_nodes[:-1]

    # save recalculating things

    q = ms ** 2 * var_x + var_y
    delta = subtract.outer(ys, cs)
    beta = (xs * var_x + (delta * ms * var_y).T).T / q
    gamma = (outer(xs, ms) - delta) ** 2 / 2 / q

    t_minus = sqrt(q / 2) / (sigma_x * sigma_y) * (x_nodes[:-1] - beta)
    t_plus = sqrt(q / 2) / (sigma_x * sigma_y) * (x_nodes[1:] - beta)

    logL = -n * LOG_2_SQRT_2PIλ
    logL += sum(log(sum(exp(-gamma) / sqrt(q) * (erf(t_plus) - erf(t_minus)), axis=-1)))

    return logL, []
