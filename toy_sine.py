"""
Toy sine example for getting my head around pypolychord.
"""

import numpy as np
import matplotlib.pyplot as plt
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior, SortedUniformPrior

try:
    from mpi4py import MPI
except ImportError:
    pass

from constants import amplitude, line_or_sine, sigma_x, sigma_y, wavelength, x_errors
from data import xs, ys
from linear_interpolation_functions import f_numpy as f


plottitle = line_or_sine
filename = "toy_" + line_or_sine

if x_errors:
    plottitle += " x errors"
    filename += "_x_errors"
    from likelihoods import x_y_errors_likelihood as likelihood
else:
    from likelihoods import y_errors_likelihood as likelihood

read_resume = False  # use this one to toggle

# Ns = np.array([2])
Ns = np.array([2, 3, 4, 5, 6])

logZs = np.zeros(len(Ns))

if __name__ == "__main__":

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
