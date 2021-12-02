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

    if not vanilla:
        plottitle += " post"
        filename += "_post"

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
        if vanilla:
            nDims = int(n_x_nodes + n_y_nodes)
        else:
            nDims = int(N * (N + 1) + 3)

        nDerived = 0

        # Define the prior (sorted uniform in x, uniform in y)

        if vanilla:

            def prior(hypercube):
                """Sorted uniform prior from Xi from [0, wavelength], unifrom prior from amplitude*[-2,2]^D for Yi."""
                return np.concatenate(
                    (
                        SortedUniformPrior(0, wavelength)(hypercube[:n_x_nodes]),
                        UniformPrior(-2 * amplitude, 2 * amplitude)(
                            hypercube[n_x_nodes:]
                        ),
                    )
                )

        else:

            def prior(hypercube):
                # start with n
                super_prior = UniformPrior(0, N)(hypercube[0:1])
                # separate off xp and yp
                theta = hypercube[1:]
                # interior nodes
                for n in np.arange(0, N) + 1:
                    start = n * (n - 1)
                    middle = n * n
                    end = n * (n + 1)
                    super_prior = np.concatenate(
                        (
                            super_prior,
                            SortedUniformPrior(0, wavelength)(theta[start:middle]),
                            UniformPrior(-2 * amplitude, 2 * amplitude)(
                                theta[middle:end]
                            ),
                        )
                    )
                # finally return with end nodes
                super_prior = np.concatenate(
                    (
                        super_prior,
                        UniformPrior(-2 * amplitude, 2 * amplitude)(hypercube[-2:]),
                    )
                )
                return super_prior

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
            likelihood,
            nDims,
            nDerived,
            settings,
            prior,
            dumper,
        )

        # | Create a paramnames file

        if vanilla:
            paramnames = [("p%i" % i, r"x_%i" % i) for i in range(n_x_nodes)]
            paramnames += [
                ("p%i" % (i + n_x_nodes), r"y_%i" % i) for i in range(n_y_nodes)
            ]

        else:
            paramnames = [("p0", "n")]
            for n in np.arange(N) + 1:
                paramnames += [
                    ("p%i" % (i + n * (n - 1) + 1), r"x%i_%i" % (i, n))
                    for i in range(n)
                ]
                paramnames += [
                    ("p%i" % (i + n * n + 1), r"y%i_%i" % (i, n)) for i in range(n)
                ]
            paramnames += [
                ("p%i" % (N * (N + 1) + 1), r"y_0"),
                ("p%i" % (N * (N + 1) + 2), r"y_%i" % (N)),
            ]

        output.make_paramnames_files(paramnames)

        # | Make an anesthetic plot (could also use getdist)

        labels = ["p%i" % i for i in range(nDims)]
        # anesthetic isn't working properly
        # from anesthetic import NestedSamples

        # samples = NestedSamples(root=settings.base_dir + "/" + settings.file_root)
        # anesthetic_fig, axes = samples.plot_2d(labels)
        # anesthetic_fig.savefig(f"plots/{filename}_anesthetic_posterior.pdf")

        # import getdist.plots

        samples, weights = samples_from_getdist_chains(
            labels, settings.base_dir + "/" + settings.file_root
        )
        sampless.append(samples)
        weightss.append(weights)
        # g = getdist.plots.getSubplotPlotter()
        # g.triangle_plot(posterior, filled=True)
        # g.export(filename + f"_{n}_posterior.pdf")

        # prior_samples = np.loadtxt(
        #     running_location.joinpath("chains/" + filename + f"_{N}_prior.txt")
        # )

        # cbar = plot_contours(
        #     f, np.linspace(0, wavelength, 100), samples, weights=weights
        # )
        # cbar = plt.colorbar(cbar, ticks=[0, 1, 2, 3], label="fgivenx")
        # cbar.set_ticklabels(["", r"$1\sigma$", r"$2\sigma$", r"$3\sigma$"])

        logZs[ii] = output.logZ

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
        plot_filename = running_location.joinpath("plots/" + filename + ".png")
    else:
        ax.legend(frameon=False)
        plot_filename = running_location.joinpath(
            "plots/" + filename + "_%i.png" % n_x_nodes
        )

    fig.savefig(plot_filename, dpi=600)

    plt.close()
    return logZs
