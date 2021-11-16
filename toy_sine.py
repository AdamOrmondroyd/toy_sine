"""
## Recreating the toy sine example from https://arxiv.org/pdf/1506.09024.pdf

Toy sine example for getting my head around pypolychord.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior, SortedUniformPrior

try:
    from mpi4py import MPI
except ImportError:
    pass

from constants import (
    amplitude,
    sigma_x,
    sigma_y,
    wavelength,
)
from data import get_data
from likelihoods import get_likelihood


def toy_sine(line_or_sine, Ns, cyclic, x_errors, read_resume=False, adam=False):
    """
    Runs polychord on the line or sine data.

    Uses piecewise linear model to compare N internal nodes for each N in Ns.

    Option for cyclic boundary conditions.
    """

    xs, ys = get_data(line_or_sine, x_errors)

    plottitle = line_or_sine
    filename = "toy_" + line_or_sine

    if x_errors:
        plottitle += " x errors"
        filename += "_x_errors"

    if cyclic:
        plottitle += " cyclic"
        filename += "_cyclic"
        if adam:
            from linear_interpolation_functions import f_cyclic_adam as f
        else:
            from linear_interpolation_functions import f_cyclic_numpy as f
    else:
        if adam:
            from linear_interpolation_functions import f_end_nodes_scipy as f
        else:
            from linear_interpolation_functions import f_end_nodes_numpy as f

    if adam:
        plottitle += " adam"
        filename += "_adam"

    likelihood = get_likelihood(line_or_sine, cyclic, x_errors)

    logZs = np.zeros(len(Ns))

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
        n_x_nodes = N
        if cyclic:
            n_y_nodes = n_x_nodes
        else:
            n_y_nodes = n_x_nodes + 2
        nDims = int(n_x_nodes + n_y_nodes)

        nDerived = 0

        # Define the prior (sorted uniform in x, uniform in y)

        def prior(hypercube):
            """Sorted uniform prior from Xi from [0, wavelength], unifrom prior from amplitude*[-2,2]^D for Yi."""
            return np.concatenate(
                (
                    SortedUniformPrior(0, wavelength)(hypercube[:n_x_nodes]),
                    UniformPrior(-2 * amplitude, 2 * amplitude)(hypercube[n_x_nodes:]),
                )
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

        paramnames = [("p%i" % i, r"x_%i" % i) for i in range(n_x_nodes)]
        paramnames += [("p%i" % (i + N), r"y_%i" % i) for i in range(n_y_nodes)]
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

        xs_to_plot = np.concatenate(([0], nodes[:n_x_nodes], [wavelength]))
        ys_to_plot = f(xs_to_plot, nodes)

        axs[0].plot(
            xs_to_plot,
            ys_to_plot,
            label="N = %i" % N,
            linewidth=0.75,
        )

        logZs[ii] = output.logZ

    axs[0].legend(frameon=False)
    axs[0].set(title=plottitle)

    if len(Ns) > 1:
        axs[1].plot(Ns, logZs, marker="+")
        axs[1].set(xlabel="N", ylabel="log(Z)")
        full_filename = Path(__file__).parent.joinpath(filename + "_comparison.png")
    else:
        full_filename = Path(__file__).parent.joinpath(filename + "_%i.png" % n_x_nodes)

    fig.savefig(full_filename, dpi=600)

    plt.close()
