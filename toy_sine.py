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
filename = "toy_sine_comparison"
read_resume = False

# if there are N nodes, we have N-2 internal nodes and 2N-4 dimensions
# change this: don't bother with end nodes, just use period wavelength in np.interp
# and don't force through origin

# Ns = np.array([2])
Ns = np.array([1, 2, 3, 4, 5, 6])
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
    ys = f(xs, [wavelength / 4, wavelength * 3 / 4, amplitude, -amplitude])

    ys += amplitude * default_rng().normal(0, sigma_y, n_points)
    return xs, ys


xs, ys = noisy_sine(n_points, amplitude, sigma_y, wavelength)

fig, axs = plt.subplots(2)

axs[0].errorbar(
    xs,
    ys,
    label="data",
    # xerr=sigma_x,
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

    def likelihood(params):
        """I suspect I'm doing this very wrong... and I haven't even started writing yet..."""
        # just x errors
        logL = -len(xs) * 0.5 * log(2 * pi * sigma_y ** 2)

        logL += np.sum(-((ys - f(xs, params)) ** 2) / 2 / sigma_y ** 2)

        # # x and y errors
        # logL = -len(xs) * log(2 * pi * sigma_x * sigma_y * wavelength)

        # def func_to_integrate(x, x_i, y_i):
        #     return e ** (
        #         -((x_i - x) ** 2) / 2 / sigma_x ** 2
        #         - (y_i - f(x, params)) ** 2 / 2 / sigma_y ** 2
        #     )

        # for i, (x_i, y_i) in enumerate(zip(xs, ys)):
        #     logL += log(quad(func_to_integrate, 0.0, wavelength, args=(x_i, y_i)))

        return logL, []

    def dumper(live, dead, logweights, logZ, logZerr):
        print("Last dead points:", dead[-1])

    # settings
    settings = PolyChordSettings(nDims, nDerived)
    settings.file_root = filename + str(N)
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

    # labels = ["p%i" % i for i in range(nDims)]

    # # | Make an anesthetic plot (could also use getdist)

    # # anesthetic isn't working properly
    # from anesthetic import NestedSamples

    # samples = NestedSamples(root=settings.base_dir + "/" + settings.file_root)
    # fig, axes = samples.plot_2d(labels)
    # fig.savefig(filename + "_anesthetic_posterior.pdf")

    # import getdist.plots

    # posterior = output.posterior
    # g = getdist.plots.getSubplotPlotter()
    # g.triangle_plot(posterior, filled=True)
    # g.export(filename + "_posterior.pdf")

    nodes = np.array(output.posterior.means)
    xs_to_plot = np.append([0], np.append(nodes[:N], [wavelength]))
    ys_to_plot = np.append([0], np.append(nodes[N:], [0]))

    logZs[ii] = output.logZ

    axs[0].plot(
        xs_to_plot,
        ys_to_plot,
        label="N = %i fit" % N,
        linewidth=0.75,
    )

axs[1].plot(Ns, logZs, marker="+")
axs[1].set(xlabel="N", ylabel="log(Z)")

axs[0].legend(frameon=False)
fig.savefig(filename + ".png", dpi=600)
plt.close()
