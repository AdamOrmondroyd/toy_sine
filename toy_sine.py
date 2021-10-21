"""
Toy sine example for getting my head around pypolychord.
"""

import numpy as np
from numpy import pi, log, sqrt, sin, abs, exp, e
from numpy.random import default_rng
import matplotlib.pyplot as plt
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior, SortedUniformPrior
from scipy.integrate import quad

try:
    from mpi4py import MPI
except ImportError:
    pass

n_points = 100
amplitude = 1.0
# sigma_x = 0.05
sigma_y = 0.05
wavelength = 1.0
filename = "toy_sine"


# linear interpolation function to fit to
def f(x, params):
    """Linear interpolation function."""
    x_nodes = params[: len(params) // 2]
    y_nodes = params[len(params) // 2 :]

    # check if x falls before the first or after the last node
    if x <= x_nodes[0]:
        return y_nodes[0] / x_nodes[0] * x
    elif x >= x_nodes[-1]:
        return (
            -y_nodes[-1] / (wavelength - x_nodes[-1]) * (x - x_nodes[-1]) + y_nodes[-1]
        )

    # find last x_node that x exceeds
    i = np.where(x_nodes <= x)[0][-1]  # where returns a tuple for some reason hence [0]
    j = i + 1

    # calculate gradient
    m = (y_nodes[j] - y_nodes[i]) / (x_nodes[j] - x_nodes[i])
    return m * (x - x_nodes[i]) + y_nodes[i]

    # should just use a scipy one of these, but good exercise
    # Will would use use np.where and np.mask
    # "numpy and array programming"
    # numpy and advanced numpy sections of "Python for data analysis"


def vectorised_f(x, params):
    """Vectorised linear interpolation function."""
    # will probably remove this but start here for sanity
    n_internal_nodes = len(params) // 2
    x_nodes = np.zeros(n_internal_nodes + 2)
    x_nodes[-1] = wavelength
    x_nodes[1:-1] = params[:n_internal_nodes]
    y_nodes = np.zeros(n_internal_nodes + 2)
    y_nodes[1:-1] = params[n_internal_nodes:]
    return np.interp(
        x,
        x_nodes,
        y_nodes,
    )


# first create noisy signal


def noisy_sine(n_points, amplitude, sigma_y, wavelength):
    """Returns tuple xs, ys where xs are uniformally distributed over a wavelength,
    and ys are sin(xs) + Gaussian noise"""
    xs = default_rng().uniform(0.0, wavelength, n_points)
    xs.sort()
    xs = xs % wavelength
    ys = amplitude * (
        sin(2 * pi / wavelength * xs) + default_rng().normal(0, sigma_y, n_points)
    )
    # xs += default_rng().normal(0, sigma_x, n_points)
    return xs, ys


def noisy_line(n_points, amplitude, sigma_y, wavelength):
    """Analogous to noisy_sine but using the "line" function from https://arxiv.org/pdf/1506.09024.pdf ."""
    xs = default_rng().uniform(0.0, wavelength, n_points)
    xs.sort()
    xs = xs % wavelength
    ys = np.zeros(len(xs))
    for i, x in enumerate(xs):
        ys[i] = f(x, [wavelength / 4, wavelength * 3 / 4, amplitude, -amplitude])

    ys += amplitude * default_rng().normal(0, sigma_y, n_points)
    return xs, ys


xs, ys = noisy_sine(n_points, amplitude, sigma_y, wavelength)


# Define the prior (sorted uniform in x, uniform in y)


def prior(hypercube):
    """Sorted uniform prior from Xi from [0, wavelength], unifrom prior from [-1,1]^D for Yi."""
    n_internal_nodes = len(hypercube) // 2
    return np.append(
        SortedUniformPrior(0, wavelength)(hypercube[:n_internal_nodes]),
        UniformPrior(-1, 1)(hypercube[n_internal_nodes:]),
    )


fig, ax = plt.subplots()
zs = np.linspace(0, wavelength, 100, endpoint=False)
fs = vectorised_f(zs, np.array([0.25, 0.75, 1, -1]))
ax.plot(zs, fs, linestyle="None", marker="+", linewidth=0.75)
ax.axhline(-1, linewidth=0.75)
ax.axhline(1, linewidth=0.75)
fig.savefig("linear interpolation function.png", dpi=600)
plt.close()


# if there are N nodes, we have N-2 internal nodes and 2N-4 dimensions
N = 4
n_internal_nodes = N - 2
nDims = 2 * n_internal_nodes
nDerived = 0


def likelihood(params):
    """I suspect I'm doing this very wrong... and I haven't even started writing yet..."""
    # logL = -len(xs) * log(2 * pi * sigma_x * sigma_y * wavelength)

    # def func_to_integrate(x, x_i, y_i):
    #     return e ** (
    #         -((x_i - x) ** 2) / 2 / sigma_x ** 2
    #         - (y_i - f(x, params)) ** 2 / 2 / sigma_y ** 2
    #     )

    logL = -len(xs) * 0.5 * log(2 * pi * sigma_y ** 2)

    for i, (xi, yi) in enumerate(zip(xs, ys)):
        logL += -((yi - f(xi, params)) ** 2) / 2 / sigma_y ** 2

    # np.vectorise is not vectorising the code, still doing a slow loop
    # need to write f so I can hand it xs

    # for i, (x_i, y_i) in enumerate(zip(xs, ys)):
    #     logL += log(quad(func_to_integrate, 0.0, wavelength, args=(x_i, y_i)))

    return logL, []


def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead points:", dead[-1])


# settings
settings = PolyChordSettings(nDims, nDerived)
settings.file_root = filename
settings.nlive = 200
settings.do_clustering = True
settings.read_resume = True

output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior, dumper)

# | Create a paramnames file

paramnames = [("p%i" % i, r"x_%i" % i) for i in range(n_internal_nodes)]
paramnames += [
    ("p%i" % (i + n_internal_nodes), r"y_%i" % i) for i in range(n_internal_nodes)
]
output.make_paramnames_files(paramnames)

labels = ["x%i" % i for i in range(n_internal_nodes)] + [
    "y%i" % i for i in range(n_internal_nodes)
]

# | Make an anesthetic plot (could also use getdist)

from anesthetic import NestedSamples

samples = NestedSamples(root=settings.base_dir + "/" + settings.file_root)
fig, axes = samples.plot_2d(labels)
fig.savefig(filename + "_anesthetic_posterior.pdf")


import getdist.plots

posterior = output.posterior
g = getdist.plots.getSubplotPlotter()
g.triangle_plot(posterior, filled=True)
g.export(filename + "_posterior.pdf")

nodes = np.array(output.posterior.means)
xs_to_plot = np.append([0], np.append(nodes[:n_internal_nodes], [wavelength]))
ys_to_plot = np.append([0], np.append(nodes[n_internal_nodes:], [0]))


fig, ax = plt.subplots()
# ax.errorbar(
#     xs, ys, xerr=sigma_x, yerr=sigma_y, linestyle="None", marker="+", linewidth=0.75
# )
ax.errorbar(
    xs,
    ys,
    label="data",
    yerr=sigma_y,
    linestyle="None",
    marker="+",
    linewidth=0.75,
    color="k",
)
ax.plot(xs_to_plot, ys_to_plot, label="fit", linewidth=0.75, color="c")
ax.axhline(-1, linewidth=0.75, color="k")
ax.axhline(1, linewidth=0.75, color="k")
ax.legend()
fig.savefig(filename + ".png", dpi=600)
plt.close()
