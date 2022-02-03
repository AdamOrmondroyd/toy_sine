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
from linear_interpolation_functions import super_model
from likelihoods import get_likelihood


def toy_sine(line_or_sine, Ns, x_errors, read_resume=False, vanilla=True):
    """
    Runs polychord on the line or sine data.

    Uses piecewise linear model to compare N internal nodes for each N in Ns.

    Option for cyclic boundary conditions.
    """

    running_location = Path(__file__).parent

    xs, ys = get_data(line_or_sine, x_errors)

    plottitle = line_or_sine
    filename = line_or_sine

    if x_errors:
        plot_filename = line_or_sine + "_x_errors/" + line_or_sine + "_x_errors"
    else:
        plot_filename = line_or_sine + "/" + line_or_sine

    if not vanilla:
        plot_filename += "_adaptive"
        plottitle += " adaptive"
        filename += "_adaptive"

    if x_errors:
        plottitle += " x errors"
        filename += "_x_errors"

    likelihood = get_likelihood(line_or_sine, x_errors, vanilla)

    logZs = np.zeros(len(Ns))
    if vanilla:
        fs = [f for i, N in enumerate(Ns)]
    else:
        fs = [super_model for i, N in enumerate(Ns)]
    sampless = []
    weightss = []

    if len(Ns) > 1:
        fig, [ax, ax_logZs] = plt.subplots(2, figsize=(6, 8))
    else:

        fig, ax = plt.subplots()

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

    for iii, N in enumerate(Ns):
        print("N = %i" % N)
        n_x_nodes = N
        n_y_nodes = n_x_nodes + 2
        if vanilla:
            nDims = int(n_x_nodes + n_y_nodes)
        else:
            nDims = int(2 * N + 3)

        nDerived = 0

        # Define the prior (sorted uniform in x, uniform in y)

        if vanilla:

            def prior(hypercube):
                """Sorted uniform prior from Xi from [0, wavelength], unifrom prior from amplitude*[-2,2]^D for Yi."""
                x_prior = SortedUniformPrior(0, wavelength)(
                    hypercube[1 : 2 * n_x_nodes + 1 : 2]
                )
                y_prior = UniformPrior(-2 * amplitude, 2 * amplitude)(
                    np.concatenate(
                        (hypercube[0 : 2 * n_x_nodes + 2 : 2], hypercube[-2:-1])
                    )
                )
                xy_prior = np.zeros(len(x_prior) + len(y_prior))
                xy_prior[1 : 2 * n_x_nodes + 1 : 2] = x_prior
                xy_prior[0 : 2 * n_x_nodes + 2 : 2] = y_prior[:-1]
                xy_prior[-1] = y_prior[-1]
                return xy_prior

        else:

            def prior(hypercube):
                n_prior = UniformPrior(0, N)(hypercube[0:1])
                x_prior = SortedUniformPrior(0, wavelength)(
                    hypercube[1 : 2 * N + 1 : 2]
                )
                y_prior = UniformPrior(-2 * amplitude, 2 * amplitude)(
                    np.concatenate((hypercube[0 : 2 * N + 2 : 2], hypercube[-1:]))
                )
                full_prior = np.zeros(
                    1 + len(x_prior) + len(y_prior), dtype=x_prior.dtype
                )
                full_prior[0] = n_prior
                full_prior[2 : 2 * N + 2 : 2] = x_prior
                full_prior[1 : 2 * N + 3 : 2] = y_prior[:-1]
                full_prior[-1] = y_prior[-1]
                return full_prior

        def dumper(live, dead, logweights, logZ, logZerr):
            print("Last dead points:", dead[-1])

        # settings
        settings = PolyChordSettings(nDims, nDerived)
        settings.file_root = filename + "_%i" % N
        settings.nlive = 25 * nDims
        settings.do_clustering = True
        settings.read_resume = read_resume

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
        if not vanilla:
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

        samples, weights = samples_from_getdist_chains(
            labels, settings.base_dir + "/" + settings.file_root
        )
        sampless.append(samples)
        weightss.append(weights)

        logZs[iii] = output.logZ

    ax.set(title=plottitle)

    cbar = plot_contours(
        fs,
        np.linspace(0, wavelength, 100),
        sampless,
        weights=weightss,
        logZ=logZs,
        ax=ax,
    )
    cbar = plt.colorbar(cbar, ticks=[0, 1, 2, 3], label="fgivenx", ax=ax)
    cbar.set_ticklabels(["", r"$1\sigma$", r"$2\sigma$", r"$3\sigma$"])

    if len(Ns) > 1:

        ax_logZs.plot(Ns, logZs, marker="+")
        ax_logZs.set(xlabel="N", ylabel="log(Z)")
        plot_filepath = running_location.joinpath("plots/" + plot_filename + ".png")
    else:
        ax.legend(frameon=False)
        plot_filepath = running_location.joinpath(
            "plots/" + plot_filename + "_%i.png" % n_x_nodes
        )

    fig.savefig(plot_filepath, dpi=600)

    plt.close()
    return logZs
