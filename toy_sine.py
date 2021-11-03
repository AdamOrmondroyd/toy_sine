"""
Toy sine example for getting my head around pypolychord.
"""

import numpy as np
from numpy import pi, log, sqrt, sin, abs, exp, subtract, outer, sum
from numpy.random import default_rng
import matplotlib.pyplot as plt
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior, SortedUniformPrior
from scipy.integrate import quad
from scipy.special import erf

try:
    from mpi4py import MPI
except ImportError:
    pass

n_points = 100
amplitude = 1.0
x_errors = True
if x_errors:
    sigma_x = 0.05
else:
    sigma_x = 0.0
sigma_y = 0.05
var_x, var_y = sigma_x ** 2, sigma_y ** 2
wavelength = 1.0
log_2_sqrt_2piλ = log(2) + 0.5 * log(2 * pi * wavelength)

line = True
if line:
    plottitle = "line"
    filename = "toy_line"
else:
    plottitle = "sine"
    filename = "toy_sine"
if x_errors:
    filename += "_x_errors"

read_resume = False  # use this one to toggle

refresh_data = True
# can't use read resume with new data
if refresh_data:
    read_resume = False

# and don't force through origin

# Ns = np.array([2])
Ns = np.array([2, 3, 4, 5, 6])

logZs = np.zeros(len(Ns))

# linear interpolation function to fit to


def f(x, params):
    """Vectorised linear interpolation function."""
    n = len(params) // 2
    return np.interp(
        x,
        params[:n],
        params[n:],
        period=wavelength,
    )


# create noisy signal


def noisy_sine(
    n_points,
    amplitude=amplitude,
    wavelength=wavelength,
    sigma_x=sigma_x,
    sigma_y=sigma_y,
):
    """Returns tuple xs, ys where xs are uniformally distributed over a wavelength,
    and ys are sin(xs) + Gaussian noise"""
    xs = default_rng().uniform(0.0, wavelength, n_points)
    xs.sort()
    ys = amplitude * (
        sin(2 * pi / wavelength * xs) + default_rng().normal(0, sigma_y, n_points)
    )
    xs += default_rng().normal(0, sigma_x, n_points)
    xs = xs % wavelength
    return xs, ys


def noisy_line(
    n_points,
    amplitude=amplitude,
    wavelength=wavelength,
    sigma_x=sigma_x,
    sigma_y=sigma_y,
):
    """Analogous to noisy_sine but using the "line" function from https://arxiv.org/pdf/1506.09024.pdf ."""
    xs = default_rng().uniform(0.0, wavelength, n_points)
    xs.sort()
    xs = xs % wavelength
    ys = np.zeros(len(xs))
    ys = f(xs, [wavelength / 4, wavelength * 3 / 4, amplitude, -amplitude])

    ys += amplitude * default_rng().normal(0, sigma_y, n_points)
    if x_errors:
        xs += default_rng().normal(0, sigma_x, n_points)
    return xs, ys


if __name__ == "__main__":

    if line:
        if refresh_data:
            xs, ys = noisy_line(n_points, amplitude, wavelength, sigma_x, sigma_y)
            np.save("line_xs.npy", xs)
            np.save("line_ys.npy", ys)
        else:
            xs = np.load("line_xs.npy")
            ys = np.load("line_ys.npy")
    else:
        if refresh_data:
            xs, ys = noisy_sine(n_points, amplitude, wavelength, sigma_x, sigma_y)
            np.save("sine_xs.npy", xs)
            np.save("sine_ys.npy", ys)
        else:
            xs = np.load("sine_xs.npy")
            ys = np.load("sine_ys.npy")

    if x_errors:

        def likelihood(params):
            """I suspect I'm doing this very wrong... and I haven't even started writing yet..."""
            # x and y errors
            n = len(params) // 2
            x_nodes = params[:n]
            y_nodes = params[n:]
            y_0 = (
                y_nodes[0]
                - (y_nodes[0] - y_nodes[-1])
                / (x_nodes[0] - (x_nodes[-1] - wavelength))
                * x_nodes[0]
            )

            x_nodes = np.append(0, np.append(x_nodes, wavelength))
            y_nodes = np.append(y_0, np.append(y_nodes, y_0))
            ms = (y_nodes[1:] - y_nodes[:-1]) / (x_nodes[1:] - x_nodes[:-1])
            cs = y_nodes[:-1] - ms * x_nodes[:-1]

            # save recalculating things

            q = ms ** 2 * var_x + var_y ** 2
            delta = subtract.outer(ys, cs)
            beta = (xs * var_x + (delta * ms * var_y).T).T / q
            gamma = (outer(xs, ms) - delta) ** 2 / 2 / q

            t_minus = sqrt(q / 2) / (sigma_x * sigma_y) * (x_nodes[:-1] - beta)
            t_plus = sqrt(q / 2) / (sigma_x * sigma_y) * (x_nodes[1:] - beta)

            logL = -n * log_2_sqrt_2piλ
            logL += sum(
                log(sum(exp(-gamma) / sqrt(q) * (erf(t_plus) - erf(t_minus)), axis=-1))
            )

            return logL, []

    else:

        def likelihood(params):
            """log likelihood function for errors in y only"""
            # just x errors
            logL = -len(xs) * 0.5 * log(2 * pi * sigma_y ** 2)
            logL += np.sum(-((ys - f(xs, params)) ** 2) / 2 / sigma_y ** 2)
            return logL, []

    if len(Ns) == 1:
        fig, axs = plt.subplots()
        axs = [axs]
    else:
        fig, axs = plt.subplots(2)

    if x_errors:
        axs[0].errorbar(
            xs,
            ys,
            label="data",
            xerr=sigma_x,
            yerr=sigma_y,
            linestyle="None",
            marker="+",
            linewidth=0.75,
            color="k",
        )
    else:
        axs[0].errorbar(
            xs,
            ys,
            label="data",
            yerr=sigma_y,
            linestyle="None",
            marker="+",
            linewidth=0.75,
            color="k",
        )
    axs[0].axhline(-1, linewidth=0.75, color="k")
    axs[0].axhline(1, linewidth=0.75, color="k")

    for ii, N in enumerate(Ns):
        print("N = %i" % N)
        nDims = int(2 * N)
        nDerived = 0

        # Define the prior (sorted uniform in x, uniform in y)

        def prior(hypercube):
            """Sorted uniform prior from Xi from [0, wavelength], unifrom prior from [-1,1]^D for Yi."""
            N = len(hypercube) // 2
            return np.append(
                SortedUniformPrior(0, wavelength)(hypercube[:N]),
                UniformPrior(-2 * amplitude, 2 * amplitude)(hypercube[N:]),
            )

        def dumper(live, dead, logweights, logZ, logZerr):
            print("Last dead points:", dead[-1])

        # settings
        settings = PolyChordSettings(nDims, nDerived)
        settings.file_root = filename + "_%i" % N
        settings.nlive = 200
        settings.do_clustering = True
        settings.read_resume = read_resume

        # run PolyChord
        output = pypolychord.run_polychord(
            likelihood, nDims, nDerived, settings, prior, dumper
        )

        # | Create a paramnames file

        paramnames = [("p%i" % i, r"x_%i" % i) for i in range(N)]
        paramnames += [("p%i" % (i + N), r"y_%i" % i) for i in range(N)]
        output.make_paramnames_files(paramnames)

        # | Make an anesthetic plot (could also use getdist)

        labels = ["p%i" % i for i in range(nDims)]
        # anesthetic isn't working properly
        from anesthetic import NestedSamples

        samples = NestedSamples(root=settings.base_dir + "/" + settings.file_root)
        fig, axes = samples.plot_2d(labels)
        fig.savefig(filename + "_anesthetic_posterior.pdf")

        # import getdist.plots

        # posterior = output.posterior
        # g = getdist.plots.getSubplotPlotter()
        # g.triangle_plot(posterior, filled=True)
        # g.export(filename + "_posterior.pdf")

        nodes = np.array(output.posterior.means)

        xs_to_plot = np.append([0], np.append(nodes[:N], [wavelength]))
        ys_to_plot = f(xs_to_plot, nodes)

        logZs[ii] = output.logZ

        axs[0].plot(
            xs_to_plot,
            ys_to_plot,
            label="N = %i" % N,
            linewidth=0.75,
        )

    axs[0].legend(frameon=False)
    axs[0].set(title=plottitle)

    if len(Ns) > 1:
        axs[1].plot(Ns, logZs, marker="+")
        axs[1].set(xlabel="N", ylabel="log(Z)")
        fig.savefig(filename + "_comparison.png", dpi=600)
    else:
        fig.savefig(filename + "_%i.png" % Ns[0], dpi=600)

    plt.close()
