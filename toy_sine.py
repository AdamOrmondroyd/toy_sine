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
from fgivenx import plot_contours, samples_from_getdist_chains

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
from linear_interpolation_functions import f_end_nodes as f
from likelihoods import get_likelihood


def toy_sine(line_or_sine, Ns, x_errors, read_resume=False):
    """
    Runs polychord on the line or sine data.

    Uses piecewise linear model to compare N internal nodes for each N in Ns.

    Option for cyclic boundary conditions.
    """

    running_location = Path(__file__).parent

    xs, ys = get_data(line_or_sine, x_errors)

    plottitle = line_or_sine
    filename = "toy_" + line_or_sine

    if x_errors:
        plottitle += " x errors"
        filename += "_x_errors"

    likelihood = get_likelihood(line_or_sine, x_errors)

    logZs = np.zeros(len(Ns))

    fig, ax = plt.subplots()

    if len(Ns) == 1:
        if x_errors:
            ax.errorbar(
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
        ax.axhline(-1, linewidth=0.75, color="k")
        ax.axhline(1, linewidth=0.75, color="k")

    for ii, N in enumerate(Ns):
        print("N = %i" % N)
        n_x_nodes = N
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
        paramnames += [("p%i" % (i + n_x_nodes), r"y_%i" % i) for i in range(n_y_nodes)]
        output.make_paramnames_files(paramnames)

        # | Make an anesthetic plot (could also use getdist)

        labels = ["p%i" % i for i in range(nDims)]
        # anesthetic isn't working properly
        from anesthetic import NestedSamples

        samples = NestedSamples(root=settings.base_dir + "/" + settings.file_root)
        fig, axes = samples.plot_2d(labels)
        fig.savefig(f"plots/{filename}_anesthetic_posterior.pdf")

        # import getdist.plots

        if len(Ns) == 1:
            samples, weights = samples_from_getdist_chains(
                labels, settings.base_dir + "/" + settings.file_root
            )
            # g = getdist.plots.getSubplotPlotter()
            # g.triangle_plot(posterior, filled=True)
            # g.export(filename + f"_{n}_posterior.pdf")

            prior_samples = np.loadtxt(
                running_location.joinpath("chains/" + filename + f"_{N}_prior.txt")
            )

            cbar = plot_contours(
                f, np.linspace(0, wavelength, 100), samples, weights=weights
            )
            cbar = plt.colorbar(cbar, ticks=[0, 1, 2, 3], label="fgivenx")
            cbar.set_ticklabels(["", r"$1\sigma$", r"$2\sigma$", r"$3\sigma$"])

        logZs[ii] = output.logZ

    ax.set(title=plottitle)

    if len(Ns) > 1:
        ax.plot(Ns, logZs, marker="+")
        ax.set(xlabel="N", ylabel="log(Z)")
        plot_filename = running_location.joinpath(
            "plots/" + filename + "_comparison.png"
        )
    else:
        ax.legend(frameon=False)
        plot_filename = running_location.joinpath(
            "plots/" + filename + "_%i.png" % n_x_nodes
        )

    fig.savefig(plot_filename, dpi=600)

    plt.close()
