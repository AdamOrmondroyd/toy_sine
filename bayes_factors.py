"""
## Recreating the toy sine example from https://arxiv.org/pdf/1506.09024.pdf

Toy sine example for getting my head around pypolychord.
"""

import numpy as np
import matplotlib.pyplot as plt

sine_pairs = [(6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7)]
line_pairs = [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7)]


def calculate_bayes_factors(logZs, pairs=sine_pairs):
    Bs = np.zeros(len(pairs))
    for i, (j, k) in enumerate(pairs):
        Bs[i] = np.mean(
            logZs[:, k - 1] - logZs[:, j - 1]
        )  # it's this way around in the paper
        sigma_Bs = np.std(logZs[:, k - 1] - logZs[:, j - 1])
    return Bs, sigma_Bs


def plot_bayes_factors(
    Bs, sigma_Bs=None, pairs=sine_pairs, line_or_sine="sine", x_errors=True
):
    fig, ax = plt.subplots()
    # if sigma_Bs is not None:
    #     ax.errorbar()

    # else:
    subscripts = [f"{{{i},{j}}}" for (i, j) in pairs]
    labels = [r"$B_{}$".format(subscript) for subscript in subscripts]
    ax.bar(
        np.arange(len(Bs)),
        Bs,
        yerr=sigma_Bs,
        color="c",
        error_kw={"capsize": 1.0, "elinewidth": 0.5, "capthick": 0.5},
    )
    ax.set_xticks(np.arange(len(Bs)))
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=-10)

    plt.savefig(f"plots/{line_or_sine}{'_x_errors' if x_errors else ''}_bayes.png")


if __name__ == "__main__":
    for line_or_sine, pairs in zip(("line", "sine"), (line_pairs, sine_pairs)):
        for x_errors in False, True:

            logZs = np.load(
                f"likelihoods/{line_or_sine}{'_x_errors' if x_errors else ''}_logZs.npy"
            )

            Bs, sigma_Bs = calculate_bayes_factors(logZs)
            plot_bayes_factors(
                Bs, sigma_Bs, pairs=pairs, line_or_sine=line_or_sine, x_errors=x_errors
            )
