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


def load_solutions(npy, dstype):
    if dstype == "map":
        md = np.load(f"{npy}/md.npy", allow_pickle=True).item()
        solns = np.memmap(
            f"{npy}/solutions.npy", dtype="float32", mode="r", shape=md["shape"]
        )
        return solns
    elif dstype == "pickle":
        d = np.load(npy, allow_pickle=True).item()
        return d["solutions"]
    else:
        raise Exception


def plot_dataset_dist(bins, npys, dstypes, labels, png, Hs, Ls, strides, spacings):
    fig, ax = plt.subplots(layout="constrained")

    width = 0.333

    # ensure descending sort for histogram fn
    bins = list(reversed(sorted(bins)))
    # ascending for plot labels
    x = np.array(list(reversed(bins[:-1])) + [bins[0] + 1])

    for i, (npy, dstype, name, H, L, stride, spacing) in enumerate(
        zip(npys, dstypes, labels, Hs, Ls, strides, spacings)
    ):
        solns = load_solutions(npy, dstype)
        solns = solns[:, ::spacing]
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
        "--dstype",
        help="dataset type",
        nargs="+",
        choices=["map", "pickle"],
        default=["map"],
    )
    parser.add_argument(
        "--labels", default=None, nargs="+", help="labels for each dataset"
    )
    parser.add_argument(
        "--H", nargs="+", default=[100], type=int, help="horizon length"
    )
    parser.add_argument("--L", nargs="+", default=[500], type=int, help="input size")
    parser.add_argument("--stride", nargs="+", default=[1], type=int, help="stride")
    parser.add_argument("--spacing", nargs="+", default=[1], type=int, help="spacing")
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
        labels = [os.path.splitext(os.path.basename(npy))[0] for npy in args.npy]

    plot_dataset_dist(
        args.bins,
        args.npy,
        args.dstype,
        labels,
        args.png,
        args.H,
        args.L,
        args.stride,
        args.spacing,
    )
