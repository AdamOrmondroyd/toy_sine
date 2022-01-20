"""
## Recreating the toy sine example from https://arxiv.org/pdf/1506.09024.pdf

Likelihood functions for fitting to a linear interpolation function with 
y errors or both x and y errors.
"""

from numpy import argsort, concatenate, log, outer, pi, subtract, sum, sqrt
from scipy.special import erf, logsumexp
from constants import sigma_x, sigma_y, wavelength
from data import get_data
from linear_interpolation_functions import f_end_nodes, get_theta_n, super_model


LOG_2_SQRT_2PIλ = log(2) + 0.5 * log(2 * pi * wavelength)
var_x, var_y = sigma_x ** 2, sigma_y ** 2


def get_likelihood(line_or_sine="sine", x_errors=True, vanilla=True):
    """Returns a likelihood function using either the "line" or "sine" data."""
    xs, ys = get_data(line_or_sine, x_errors)
    xs_sorted_index = argsort(xs)
    xs, ys = xs[xs_sorted_index], ys[xs_sorted_index]

    if x_errors:

        def x_y_errors_end_nodes_likelihood(params):
            n = len(params) // 2 - 1
            x_nodes = concatenate(([0], params[1 : 2 * n + 1 : 2], [wavelength]))
            y_nodes = concatenate((params[0 : 2 * n + 2 : 2], params[-1:]))

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
            logL = sum(
                logsumexp(
                    -gamma + log(q ** -0.5 * (erf(t_plus) - erf(t_minus))), axis=-1
                )
            )

            return logL, []

        if vanilla:

            return x_y_errors_end_nodes_likelihood

        else:

            def super_likelihood(params):

                theta_n = get_theta_n(params)
                return x_y_errors_end_nodes_likelihood(theta_n)

            return super_likelihood

    else:

        if vanilla:

            def y_errors_end_nodes_likelihood(params):
                logL = -len(ys) * 0.5 * log(2 * pi * var_y)
                logL += sum(-((ys - f_end_nodes(xs, params)) ** 2) / 2 / var_y)
                return logL, []

            return y_errors_end_nodes_likelihood
        else:

            def y_errors_end_nodes_supermodel_likelihood(params):
                logL = -len(ys) * 0.5 * log(2 * pi * var_y)
                logL += sum(-((ys - super_model(xs, params)) ** 2) / 2 / var_y)
                return logL, []

            return y_errors_end_nodes_supermodel_likelihood
