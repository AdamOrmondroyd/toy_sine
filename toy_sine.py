"""
## Recreating the toy sine example from https://arxiv.org/pdf/1506.09024.pdf

Toy sine example for getting my head around pypolychord.
"""

from pathlib import Path
from selectors import EpollSelector
import numpy as np
import matplotlib.pyplot as plt
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior, SortedUniformPrior
from fgivenx import plot_contours, samples_from_getdist_chains
from linf import Linf, AdaptiveLinf, LinfPrior, AdaptiveLinfPrior, LinfLikelihood

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


def toy_sine(
    line_or_sine, Ns, x_errors, read_resume=False, adaptive=False, repeat_num=None
):
    """
    Runs polychord on the line or sine data.

    Uses piecewise linear model to compare N internal nodes for each N in Ns.

    Option for cyclic boundary conditions.
    """

    filename = line_or_sine

    if adaptive:
        filename += "_adaptive"

    if x_errors:
        filename += "_x_errors"

    x_data, y_data = get_data(line_or_sine, x_errors)

    if x_errors:
        sigma = np.array([sigma_x, sigma_y])
    else:
        sigma = sigma_y

    likelihood = LinfLikelihood(0, wavelength, x_data, y_data, sigma, adaptive)

    logZs = np.zeros(len(Ns))

    for iii, N in enumerate(Ns):
        print("N = %i" % N)
        n_x_nodes = N - 2
        n_y_nodes = N
        nDims = int(n_x_nodes + n_y_nodes)
        if adaptive:
            nDims += 1

        nDerived = 0

        # Define the prior (sorted uniform in x, uniform in y)

        if adaptive:

            prior = AdaptiveLinfPrior(0, wavelength, -2 * amplitude, 2 * amplitude, N)

        else:

            prior = LinfPrior(0, wavelength, -2 * amplitude, 2 * amplitude)

        def dumper(live, dead, logweights, logZ, logZerr):
            print("Last dead points:", dead[-1])

        # settings
        settings = PolyChordSettings(nDims, nDerived)
        settings.file_root = filename + f"_{N}"
        if repeat_num is not None:
            settings.file_root += f"_{repeat_num}"
        settings.nlive = 25 * nDims
        settings.do_clustering = True
        settings.read_resume = read_resume
        settings.num_repeats = 5 * nDims

        # run PolyChord
        output = pypolychord.run_polychord(
            likelihood,
            nDims,
            nDerived,
            settings,
            prior,
            dumper,
        )

        # | Create a paramnames file

        paramnames = []
        i = 0
        if adaptive:
            paramnames += [("p%i" % i, "n")]
            i += 1

        for ii in range(n_x_nodes):
            paramnames += [("p%i" % i, r"y_%i" % ii)]
            i += 1
            paramnames += [("p%i" % i, r"x_%i" % (ii + 1))]
            i += 1
        paramnames += [
            ("p%i" % i, r"y_%i" % (ii + 1)),
            ("p%i" % (i + 1), r"y_%i" % (ii + 2)),
        ]

        output.make_paramnames_files(paramnames)

        logZs[iii] = output.logZ

    np.save(
        f'likelihoods/{line_or_sine}{"_x_errors" if x_errors else ""}{"_adaptive" if adaptive else ""}_logZs.npy',
        logZs,
    )

    return logZs


def plot_toy_sine(line_or_sine, N, x_errors, adaptive=False, show=False, ax=None):
    """
    Plot the results from toy_sine()
    """
    logZs = np.load(
        f'likelihoods/{line_or_sine}{"_x_errors" if x_errors else ""}{"_adaptive" if adaptive else ""}_logZs.npy'
    )

    running_location = Path(__file__).parent

    plottitle = line_or_sine
    if x_errors:
        plot_filename = line_or_sine + "_x_errors/" + line_or_sine + "_x_errors"
    else:
        plot_filename = line_or_sine + "/" + line_or_sine

    if adaptive:
        plot_filename += "_adaptive"
        plottitle += " adaptive"

    if x_errors:
        plottitle += " x errors"

    if adaptive:
        fs = AdaptiveLinf(0, wavelength)
    else:
        fs = Linf(0, wavelength)

    if ax:
        show = False
        save = False
    else:
        save = True
        fig, ax = plt.subplots()

    xs, ys = get_data(line_or_sine, x_errors)
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
    print("N = %i" % N)
    n_x_nodes = N - 2
    n_y_nodes = N
    nDims = n_x_nodes + n_y_nodes
    if adaptive:
        nDims += 1

    # | Make an anesthetic plot (could also use getdist)

    labels = ["p%i" % i for i in range(nDims)]
    # anesthetic isn't working properly
    # from anesthetic import NestedSamples

    # samples = NestedSamples(root=settings.base_dir + "/" + settings.file_root)
    # anesthetic_fig, axes = samples.plot_2d(labels)
    # anesthetic_fig.savefig(f"plots/{plot_filename}_{N}_anesthetic_posterior.pdf")
    # if not vanilla:
    #     n_fig, n_axes = samples.plot_1d(["p0"], plot_type="hist")
    #     n_fig.savefig(f"plots/{plot_filename}_{N}_n_posterior.png")

    # import getdist.plots
    chains_path = "chains/"
    chains_path += line_or_sine

    if adaptive:
        chains_path += "_adaptive"

    if x_errors:
        chains_path += "_x_errors"
    chains_path += f"_{N}"

    samples, weights = samples_from_getdist_chains(labels, chains_path)

    ax.set(title=plottitle)

    cbar = plot_contours(
        fs,
        np.linspace(0, wavelength, 100),
        samples,
        weights=weights,
        # logZ=logZs,
        ax=ax,
    )
    cbar = plt.colorbar(cbar, ticks=[0, 1, 2, 3], label="fgivenx", ax=ax)
    cbar.set_ticklabels(["", r"$1\sigma$", r"$2\sigma$", r"$3\sigma$"])

    # if len(Ns) > 1:

    #     ax_logZs.plot(Ns, logZs, marker="+")
    #     ax_logZs.set(xlabel="N", ylabel="log(Z)")
    #     plot_filepath = running_location.joinpath("plots/" + plot_filename + ".png")
    # else:
    ax.legend(frameon=False)
    plot_filepath = running_location.joinpath(
        "plots/" + plot_filename + "_%i_linf.png" % n_x_nodes
    )
    if save:
        fig.savefig(plot_filepath, dpi=600)
    if show:
        plt.show()

    plt.close()
