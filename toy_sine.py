"""
## Recreating the toy sine example from https://arxiv.org/pdf/1506.09024.pdf

Toy sine example for getting my head around pypolychord.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pypolychord
from pypolychord.settings import PolyChordSettings
from pymultinest.solve import solve

# from fgivenx import plot_contours, samples_from_getdist_chains
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
from paramnames import paramnames_file

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


def cluster(
    position_matrix,
):
    print("started clustering")
    amount_initial_centers = 1
    initial_centers = kmeans_plusplus_initializer(
        position_matrix, amount_initial_centers
    ).initialize()

    xmeans_instance = xmeans(position_matrix, initial_centers, 8, ccore=False)
    xmeans_instance.process()
    clusters = xmeans_instance.get_clusters()
    cluster_list = np.zeros(len(position_matrix))
    for i, cluster in enumerate(clusters):
        cluster_list[cluster] = i
    print("finished clustering")
    return cluster_list


def toy_sine(
    line_or_sine,
    N,
    x_errors,
    read_resume=False,
    adaptive=False,
    repeat_num=None,
    use_multinest=False,
    use_xmeans=False,
):
    """
    Runs polychord on the line or sine data.

    fits a linf to either the line or sine data.
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

    print("N = %i" % N)
    n_x_nodes = max(N - 2, 0)
    n_y_nodes = N
    nDims = int(n_x_nodes + n_y_nodes)
    if adaptive:
        nDims += 1

    nDerived = 0

    # Define the prior (sorted uniform in x, uniform in y)

    if adaptive:

        prior = AdaptiveLinfPrior(
            0, wavelength, -1.5 * amplitude, 1.5 * amplitude, 1, N
        )

    else:

        prior = LinfPrior(0, wavelength, -1.5 * amplitude, 1.5 * amplitude)

    def dumper(live, dead, logweights, logZ, logZerr):
        print("Last dead points:", dead[-1])

    # settings
    settings = PolyChordSettings(nDims, nDerived)
    settings.file_root = filename + f"_{N}"
    if use_xmeans:
        settings.file_root += "_xmeans"
    if repeat_num is not None:
        settings.file_root += f"_{repeat_num}"
    settings.nlive = 25 * nDims
    settings.do_clustering = True
    settings.read_resume = read_resume
    # settings.num_repeats = 5 * nDims
    if use_multinest:
        multihood = lambda theta: likelihood(theta)[0]
        solve(
            LogLikelihood=multihood,
            Prior=prior,
            n_dims=nDims,
            outputfiles_basename="multinest_chains/" + settings.file_root,
            resume=read_resume,
            verbose=True,
            n_live_points=25 * nDims,
        )
    else:
        # run PolyChord
        if use_xmeans:
            output = pypolychord.run_polychord(
                likelihood,
                nDims,
                nDerived,
                settings,
                prior,
                dumper,
                cluster,
            )
        else:
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
    if adaptive:
        paramnames += [("N", "N")]

    if n_y_nodes > 1:
        paramnames += [(f"y0", f"y_0")]
    for ii in range(n_x_nodes):
        paramnames += [(f"x{ii+1}", f"x_{ii+1}")]
        paramnames += [(f"y{ii+1}", f"y_{ii+1}")]
    paramnames += [(f"y{N-1}", f"y_{N-1}")]

    if use_multinest:
        paramnames_file("multinest_chains/" + settings.file_root, paramnames)
    else:
        output.make_paramnames_files(paramnames)

    return


# def plot_toy_sine(line_or_sine, N, x_errors, adaptive=False, show=False, ax=None):
#     """
#     Plot the results from toy_sine()
#     """
#     logZs = np.load(
#         f'likelihoods/{line_or_sine}{"_x_errors" if x_errors else ""}{"_adaptive" if adaptive else ""}_logZs.npy'
#     )

#     running_location = Path(__file__).parent

#     plottitle = line_or_sine
#     if x_errors:
#         plot_filename = line_or_sine + "_x_errors/" + line_or_sine + "_x_errors"
#     else:
#         plot_filename = line_or_sine + "/" + line_or_sine

#     if adaptive:
#         plot_filename += "_adaptive"
#         plottitle += " adaptive"

#     if x_errors:
#         plottitle += " x errors"

#     if adaptive:
#         fs = AdaptiveLinf(0, wavelength)
#     else:
#         fs = Linf(0, wavelength)

#     if ax:
#         show = False
#         save = False
#     else:
#         save = True
#         fig, ax = plt.subplots()

#     xs, ys = get_data(line_or_sine, x_errors)
#     if x_errors:
#         ax.errorbar(
#             xs,
#             ys,
#             label="data",
#             xerr=sigma_x,
#             yerr=sigma_y,
#             linestyle="None",
#             marker="+",
#             linewidth=0.75,
#        chains_path = "chains/"
#     chains_path += line_or_sine

#     if adaptive:
#         chains_path += "_adaptive"

#     if x_errors:
#         chains_path += "_x_errors"
#     chains_path += f"_{N}"

#     samples, weights = samples_from_getdist_chains(labels, chains_path)

#     ax.set(title=plottitle)

#     cbar = plot_contours(
#         fs,
#         np.linspace(0, wavelength, 100),
#         samples,
#         weights=weights,
#         # logZ=logZs,
#         ax=ax,
#     )
#     cbar = plt.colorbar(cbar, ticks=[0, 1, 2, 3], label="fgivenx", ax=ax)
#     cbar.set_ticklabels(["", r"$1\sigma$", r"$2\sigma$", r"$3\sigma$"])

#     # if len(Ns) > 1:

#     #     ax_logZs.plot(Ns, logZs, marker="+")
#     #     ax_logZs.set(xlabel="N", ylabel="log(Z)")
#     #     plot_filepath = running_location.joinpath("plots/" + plot_filename + ".png")
#     # else:
#     ax.legend(frameon=False)
#     plot_filepath = running_location.joinpath(
#         "plots/" + plot_filename + "_%i_linf.png" % n_x_nodes
#     )
#     if save:
#         fig.savefig(plot_filepath, dpi=600)
#     if show:
#         plt.show()

#     plt.close()
#        color="k",
#         )
#     else:
#         ax.errorbar(
#             xs,
#             ys,
#             label="data",
#             yerr=sigma_y,
#             linestyle="None",
#             marker="+",
#             linewidth=0.75,
#             color="k",
#         )
#     ax.axhline(-1, linewidth=0.75, color="k")
#     ax.axhline(1, linewidth=0.75, color="k")
#     print("N = %i" % N)
#     n_x_nodes = N - 2
#     n_y_nodes = N
#     nDims = n_x_nodes + n_y_nodes
#     if adaptive:
#         nDims += 1

#     # | Make an anesthetic plot (could also use getdist)

#     labels = ["p%i" % i for i in range(nDims)]
#     # anesthetic isn't working properly
#     # from anesthetic import NestedSamples

#     # samples = NestedSamples(root=settings.base_dir + "/" + settings.file_root)
#     # anesthetic_fig, axes = samples.plot_2d(labels)
#     # anesthetic_fig.savefig(f"plots/{plot_filename}_{N}_anesthetic_posterior.pdf")
#     # if not vanilla:
#     #     n_fig, n_axes = samples.plot_1d(["p0"], plot_type="hist")
#     #     n_fig.savefig(f"plots/{plot_filename}_{N}_n_posterior.png")

#     # import getdist.plots
