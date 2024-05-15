import numpy as np
import matplotlib.pyplot as plt
from functools import cmp_to_key
from utils import get_local_minima_from_solutions


def hist_local_minima(solns, H, L, stride, bins):
    nseries, npts, ndim = solns.shape
    win_per_srs = (npts - L - H) // stride + 1
    hist = {b: 0 for b in bins[:-1]}
    minima, mindex = get_local_minima_from_solutions(solns[:, L:])
    count = 0
    for bidx in range(len(bins) - 1):
        upper = bins[bidx]
        lower = bins[bidx + 1]
        binmembers = (minima <= upper) & (minima > lower)
        n = np.minimum(H, npts - mindex[binmembers][:, 1] + L + 1).sum() // stride
        hist[upper] += n
        count += n
    hist[float("inf")] = nseries * win_per_srs - count
    return hist


def plot_dataset_dist(bins, npys, labels, png, Hs, Ls, strides, spacings):
    fig, ax = plt.subplots(layout="constrained")

    width = 0.333

    # ensure descending sort for histogram fn
    bins = list(reversed(sorted(bins)))
    # ascending for plot labels
    x = np.array(list(reversed(bins[:-1])) + [bins[0] + 1])

    for i, (npy, name, H, L, stride, spacing) in enumerate(
        zip(npys, labels, Hs, Ls, strides, spacings)
    ):
        d = np.load(npy, allow_pickle=True).item()
        solns = d["solutions"][:, ::spacing]
        hist = hist_local_minima(solns, H, L, stride, bins)
        pairs = sorted(hist.items(), key=cmp_to_key(lambda a, b: a[0] - b[0]))
        offset = width * i
        heights = [n for _, n in pairs]
        rects = ax.bar(x + offset, heights, width, label=name)
        ax.bar_label(rects, fmt="{:.0e}", fontsize=6)

    ax.set_title("Dataset Distributions")
    ax.set_yscale("log")
    ax.set_ylabel("number of windows")
    xlabels = [f"<{xv}" for xv in x[:-1]] + [f">{bins[0]}"]
    ax.set_xticks(x + width / len(npys), xlabels, fontsize=8)
    ax.set_xlabel("minimum distance from origin")
    ax.legend(loc="upper left", ncols=2)
    plt.savefig(png, dpi=500)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("npy", help="dataset file(s)", nargs="+")
    parser.add_argument("png", help="output png filename")
    parser.add_argument(
        "--labels", default=None, nargs="+", help="labels for each dataset"
    )
    parser.add_argument("--H", nargs="+", default=100, type=int, help="horizon length")
    parser.add_argument("--L", nargs="+", default=500, type=int, help="input size")
    parser.add_argument("--stride", nargs="+", default=1, type=int, help="stride")
    parser.add_argument("--spacing", nargs="+", default=1, type=int, help="spacing")
    parser.add_argument(
        "--bins",
        nargs="+",
        type=int,
        default=[5, 4, 3, 2, 1, 0],
        help="histogram bin edges",
    )
    args = parser.parse_args()

    labels = args.labels
    if labels is None:
        labels = os.path.splitext(os.path.basename(args.npy))[0]

    plot_dataset_dist(
        args.bins, args.npy, labels, args.png, args.H, args.L, args.stride, args.spacing
    )
