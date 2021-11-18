"""
## Recreating the toy sine example from https://arxiv.org/pdf/1506.09024.pdf

Likelihood functions for fitting to a linear interpolation function with 
y errors or both x and y errors.
"""

from numpy import argsort, concatenate, log, outer, pi, subtract, sum, sqrt
from scipy.special import erf, logsumexp
from constants import sigma_x, sigma_y, wavelength
from data import get_data
from linear_interpolation_functions import (
    f_end_nodes_numpy,
    f_cyclic_numpy,
    f_cyclic_adam,
)


LOG_2_SQRT_2PIλ = log(2) + 0.5 * log(2 * pi * wavelength)
var_x, var_y = sigma_x ** 2, sigma_y ** 2


def get_likelihood(line_or_sine="sine", cyclic=False, x_errors=True, adam=False):
    """Returns a likelihood function using either the "line" or "sine" data."""
    xs, ys = get_data(line_or_sine, x_errors)
    xs_sorted_index = argsort(xs)
    xs, ys = xs[xs_sorted_index], ys[xs_sorted_index]

    if cyclic:
        if x_errors:

            def x_y_errors_cyclic_likelihood(params):
                n = len(params) // 2
                x_nodes = params[:n]
                y_nodes = params[n:]
                y_0 = (
                    y_nodes[0]
                    - (y_nodes[0] - y_nodes[-1])
                    / (x_nodes[0] - (x_nodes[-1] - wavelength))
                    * x_nodes[0]
                )

                x_nodes = concatenate(([0], x_nodes, [wavelength]))
                y_nodes = concatenate(([y_0], y_nodes, [y_0]))
                ms = (y_nodes[1:] - y_nodes[:-1]) / (x_nodes[1:] - x_nodes[:-1])
                cs = y_nodes[:-1] - ms * x_nodes[:-1]

                # save recalculating things

                q = ms ** 2 * var_x + var_y
                delta = subtract.outer(ys, cs)
                beta = (xs * var_x + (delta * ms * var_y).T).T / q
                gamma = (outer(xs, ms) - delta) ** 2 / 2 / q

                t_minus = sqrt(q / 2) / (sigma_x * sigma_y) * (x_nodes[:-1] - beta)
                t_plus = sqrt(q / 2) / (sigma_x * sigma_y) * (x_nodes[1:] - beta)

                logL = -len(xs) * LOG_2_SQRT_2PIλ
                logL += sum(
                    logsumexp(-gamma + log(q ** -0.5 * (erf(t_plus) - erf(t_minus))))
                )

                return logL, []

            return x_y_errors_cyclic_likelihood

        else:

            def y_errors_cyclic_likelihood(params):
                logL = -len(ys) * 0.5 * log(2 * pi * var_y)
                if adam:
                    logL += sum(-((ys - f_cyclic_adam(xs, params)) ** 2) / 2 / var_y)
                else:
                    logL += sum(-((ys - f_cyclic_numpy(xs, params)) ** 2) / 2 / var_y)
                return logL, []

            return y_errors_cyclic_likelihood

    else:
        if x_errors:

            def x_y_errors_end_nodes_likelihood(params):
                n = len(params) // 2 - 1
                x_nodes = params[:n]
                x_nodes = concatenate(([0], x_nodes, [wavelength]))
                y_nodes = params[n:]

                ms = (y_nodes[1:] - y_nodes[:-1]) / (x_nodes[1:] - x_nodes[:-1])
                cs = y_nodes[:-1] - ms * x_nodes[:-1]

                # save recalculating things

                q = ms ** 2 * var_x + var_y
                delta = subtract.outer(ys, cs)
                beta = (xs * var_x + (delta * ms * var_y).T).T / q
                gamma = (outer(xs, ms) - delta) ** 2 / 2 / q

                t_minus = sqrt(q / 2) / (sigma_x * sigma_y) * (x_nodes[:-1] - beta)
                t_plus = sqrt(q / 2) / (sigma_x * sigma_y) * (x_nodes[1:] - beta)

                logL = -len(xs) * LOG_2_SQRT_2PIλ
                logL += sum(
                    logsumexp(-gamma + log(q ** -0.5 * (erf(t_plus) - erf(t_minus))))
                )

                return logL, []

            return x_y_errors_end_nodes_likelihood

        else:

            def y_errors_end_nodes_likelihood(params):
                logL = -len(ys) * 0.5 * log(2 * pi * var_y)
                logL += sum(-((ys - f_end_nodes_numpy(xs, params)) ** 2) / 2 / var_y)
                return logL, []

            return y_errors_end_nodes_likelihood
