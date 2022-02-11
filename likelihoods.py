"""
## Recreating the toy sine example from https://arxiv.org/pdf/1506.09024.pdf

Likelihood functions for fitting to a linear interpolation function with 
y errors or both x and y errors.
"""

import numpy as np
from scipy.special import erf, logsumexp
from constants import sigma_x, sigma_y, wavelength
from data import get_data
from linear_interpolation_functions import f_end_nodes, get_theta_n, super_model


LOG_2_SQRT_2PIλ = np.log(2) + 0.5 * np.log(2 * np.pi * wavelength)
var_x, var_y = sigma_x ** 2, sigma_y ** 2


def get_likelihood(line_or_sine="sine", x_errors=True, adaptive=False):
    """Returns a likelihood function using either the "line" or "sine" data."""
    xs, ys = get_data(line_or_sine, x_errors)

    if x_errors:

        def x_y_errors_end_nodes_likelihood(params):
            n = len(params) // 2 - 1
            x_nodes = np.concatenate(([0], params[1 : 2 * n + 1 : 2], [wavelength]))
            y_nodes = np.concatenate((params[0 : 2 * n + 2 : 2], params[-1:]))

            ms = (y_nodes[1:] - y_nodes[:-1]) / (x_nodes[1:] - x_nodes[:-1])
            cs = y_nodes[:-1] - ms * x_nodes[:-1]

            # save recalculating things

            q = (np.outer(var_x, ms ** 2).T + var_y).T
            delta = np.subtract.outer(ys, cs)
            beta = (xs * var_x + (delta * ms).T * var_y).T / q
            gamma = (np.outer(xs, ms) - delta) ** 2 / 2 / q

            t_minus = (np.sqrt(q / 2).T / (sigma_x * sigma_y)).T * (x_nodes[:-1] - beta)
            t_plus = (np.sqrt(q / 2).T / (sigma_x * sigma_y)).T * (x_nodes[1:] - beta)

            logL = -len(xs) * LOG_2_SQRT_2PIλ
            logL = np.sum(
                logsumexp(
                    -gamma + np.log(q ** -0.5 * (erf(t_plus) - erf(t_minus))), axis=-1
                )
            )

            return logL, []

        if not adaptive:

            return x_y_errors_end_nodes_likelihood

        else:

            def super_likelihood(params):

                theta_n = get_theta_n(params)
                return x_y_errors_end_nodes_likelihood(theta_n)

            return super_likelihood

    else:

        if adaptive:
            f = super_model

        else:
            f = f_end_nodes

        def y_errors_end_nodes_likelihood(params):
            if hasattr(var_y, "__len__"):
                logL = -0.5 * np.sum(np.log(2 * np.pi * var_y))
            else:
                logL = -0.5 * len(ys) * np.log(2 * np.pi * var_y)

            logL += np.sum(-((ys - f(xs, params)) ** 2) / 2 / var_y)
            return logL, []

        return y_errors_end_nodes_likelihood
