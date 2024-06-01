import multiprocessing as mp
import os
from functools import partial
from collections import defaultdict

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from data_generation.generate_dataset import PERIOD, solve_lorenz
from data_generation.utils import get_local_minima_from_solutions

plt.rcParams["keymap.back"].remove("left")
plt.rcParams["keymap.forward"].remove("right")
plt.rcParams["keymap.pan"].remove("p")
plt.rcParams["keymap.quit"].remove("q")
plt.rcParams["keymap.fullscreen"].remove("f")

critical_points = np.array([[0, 0, 0], [8.49, 8.49, 27], [-8.49, -8.49, 27]])


def calc_acc_dist(yh, yt, err_thresh):
    nseries, npts, _ = yt.shape
    flags = np.array([False] * nseries)
    scores = np.zeros(nseries)
    errs = np.linalg.norm(yh - yt, axis=-1)
    for i in range(npts):
        stop_idx = np.argwhere(~flags & (errs[:, i] > err_thresh))
        scores[stop_idx] = i
        flags = flags | (errs[:, i] > err_thresh)

    return scores


def plot_compare_full(yt, yh, ncomp, npts):
    fig = plt.figure()
    series = np.random.choice(yt.shape[0], ncomp, replace=False)

    gs = gridspec.GridSpec(2, ncomp)
    gs.update(wspace=0, hspace=0)

    def plot(data, idx, color):
        ax = fig.add_subplot(gs[ncomp * idx + i], projection="3d")
        ax.plot(*data.T, color=color)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        if i == ncomp // 2:
            ax.set_title("IVP Solver" if idx == 0 else "Model 3")

    for i, s in enumerate(series):
        plot(yt[s, :npts], 0, "C0")
        plot(yh[s, :npts], 1, "C1")
    fig.subplots_adjust(bottom=0, top=0.9)


def plot_3d(yh, yt, coords, winsize, dt, minimal=False):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.axes.set_xlim3d(left=-20, right=20)
    ax.axes.set_ylim3d(bottom=-20, top=20)
    ax.axes.set_zlim3d(bottom=-5, top=50)
    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.view_init(elev=15, azim=-45)
    ax.scatter(*critical_points.T, label="critical points", color="purple")

    sidx = 0
    zcrit = None
    if sidx in coords:
        zs = np.array([yh[sidx, zidx] for zidx in coords[sidx]])
        zcrit = ax.scatter(*zs.T, label="z-critical", color="lime")

    startidx = 0
    endidx = startidx + winsize

    ytwin = yt[sidx, startidx:endidx]
    yt3d = ax.plot(*ytwin.T, label="reference", alpha=1)[0]
    yhwin = yh[sidx, startidx:endidx]
    yh3d = ax.plot(*yhwin.T, label="prediction", alpha=1)[0]

    if not minimal:
        ax.legend(
            loc="upper right", bbox_to_anchor=(1, 0.9), bbox_transform=fig.transFigure
        )

    err = np.linalg.norm(ytwin - yhwin, axis=1).max()
    dfo = np.linalg.norm(ytwin, axis=1).min()

    def title(sidx, endidx, err, dfo):
        if minimal:
            ax.set_title(f"series {sidx}, time = {dt*endidx:.2f}", y=0.85, x=0.5)
        else:
            ax.set_title(
                f"""
                                series {sidx}
                
                      time = {dt*endidx:.2f},    max L2 diff = {err:.2f} dfo = {dfo:.2f}
                """,
                y=0.9,
                x=0.2,
            )

    title(sidx, endidx, err, dfo)
    btm = -0.1
    top = 0.9
    if minimal:
        top = 1.1
    fig.subplots_adjust(left=-0.1, right=1.1, bottom=btm, top=top)

    def onkeypress(yh, yt, fig, ax, state, e):
        sidx = state["sidx"]
        startidx = state["startidx"]
        endidx = state["endidx"]
        inc = state["increment"]
        winsize = state["winsize"]
        nseries, npts, ndim = yh.shape

        def update_plot(sidx, startidx, endidx):
            yhwin = yh[sidx, startidx:endidx]
            state["yh3d"].set_data_3d(*yhwin.T)
            ytwin = yt[sidx, startidx:endidx]
            state["yt3d"].set_data_3d(*ytwin.T)

            err = np.linalg.norm(ytwin - yhwin, axis=1).max()
            dfo = np.linalg.norm(ytwin, axis=1).min()
            title(sidx, endidx, err, dfo)
            state["sidx"] = sidx
            state["startidx"] = startidx
            state["endidx"] = endidx

            fig.canvas.draw_idle()

        if e.key in ["b", "f", "B", "F"]:
            if e.key == "B":
                startidx = min(endidx - winsize, startidx + winsize)
            elif e.key == "b":
                startidx = max(0, startidx - winsize)
            elif e.key == "F":
                endidx = max(startidx + winsize, endidx - inc)
            else:
                endidx = min(endidx + inc, yh.shape[1])

            update_plot(sidx, startidx, endidx)

        elif e.key in ["z", "a"]:
            sign = 1 if e.key == "z" else -1
            state["coordidx"] = (state["coordidx"] + len(coords[sidx]) + sign) % len(
                coords[sidx]
            )
            zidx = coords[sidx][state["coordidx"]]
            winlen = endidx - startidx
            startidx = max(0, zidx - winsize // 2)
            endidx = startidx + winlen
            update_plot(sidx, startidx, endidx)

        elif e.key in ["p", "q"]:
            if e.key == "p":
                sidx = (sidx + 1) % nseries
                endidx -= startidx
                startidx = 0
            elif e.key == "q":
                sidx = (sidx + nseries - 1) % nseries
                startidx = npts - (endidx - startidx)
                endidx = npts
            state["zcrit"].remove()
            zs = np.array([yh[sidx, zidx] for zidx in coords[sidx]])
            state["zcrit"] = ax.scatter(*zs.T, label="z-critical", color="lime")
            update_plot(sidx, startidx, endidx)

        elif e.key == "h":
            state["yt3d"].set_visible(not state["yt3d"].get_visible())
            fig.canvas.draw_idle()
            ax.set_title("")

        elif any([k in e.key for k in ["left", "right"]]):
            if "shift" in e.key:
                inc *= 10

            if "left" in e.key:
                if startidx >= inc:
                    startidx -= inc
                    endidx -= inc
                else:
                    sidx = (nseries + sidx - 1) % nseries
                    startidx = npts - (endidx - startidx)
                    endidx = npts
            elif "right" in e.key:
                if endidx <= npts - winsize - inc:
                    startidx += inc
                    endidx += inc
                else:
                    sidx = (sidx + 1) % nseries
                    endidx -= startidx
                    startidx = 0

            update_plot(sidx, startidx, endidx)

    fig.canvas.mpl_connect(
        "key_press_event",
        partial(
            onkeypress,
            yh,
            yt,
            fig,
            ax,
            {
                "zcrit": zcrit,
                "coordidx": 0,
                "sidx": sidx,
                "startidx": startidx,
                "endidx": endidx,
                "winsize": winsize,
                "yh3d": yh3d,
                "yt3d": yt3d,
                "increment": 1,
            },
        ),
    )


def lorenz_ivp(
    N,
    ppp,
    methods=["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"],
    ivs=np.array([[-9.7869288, -15.03852, 20.533978]]),
):
    results = {}
    with mp.Pool(processes=os.cpu_count()) as pool:
        for m in methods:
            func = partial(solve_lorenz, N, ppp, m)
            ys = []
            for y in tqdm.tqdm(pool.imap(func, ivs), total=len(ivs)):
                ys.append(y)
            results[m] = {"y": np.stack(ys)}
    return results


def compare_ivp(ytrue, yhat, dt):
    N = ytrue.shape[1]
    ivs = ytrue[:, 0]
    ppp = int(PERIOD / dt)
    res = lorenz_ivp(N, ppp, methods=["RK45", "Radau"], ivs=ivs)
    rk45 = res["RK45"]["y"]
    radau = res["Radau"]["y"]

    mae = np.abs(ytrue - yhat).mean(axis=(0, 2))
    ivp_mae = np.abs(radau - rk45).mean(axis=(0, 2))
    ax = plt.figure().add_subplot()
    t = np.linspace(0, (N - 1) * dt, N)
    ax.plot(t, mae, label="Radau vs NHiTS")
    ax.plot(t, ivp_mae, label="Radau vs RK45")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("MSE")
    ax.legend()


def print_minima(yhat, dt, nprint=10):
    minima, mindex = get_local_minima_from_solutions(yhat)
    sortidx = np.argsort(minima)
    print(f"Top {nprint} local minima:")
    for i in range(nprint):
        idx = sortidx[i]
        print(
            f"\t{minima[idx]:.2f} @ Series {mindex[idx][0]}, {dt*mindex[idx][1]:.2f}s"
        )


def collect_z_traj(yhat, zmin, zmax):
    candidx = np.argwhere((yhat[:, :, 2] >= zmin) & (yhat[:, :, 2] <= zmax))
    indices = defaultdict(list)
    npts = yhat.shape[1]
    for s, i in candidx:
        if (
            i > 0
            and i < npts - 1
            and yhat[s, i, 2] > yhat[s, i - 1, 2]
            and yhat[s, i, 2] > yhat[s, i + 1, 2]
        ):
            indices[s].append(i)
    return indices


def statistics(model, solver, dt):
    def exp_k2(y, k=2):
        return np.exp(-(y[:, :, k].astype(np.float64) ** 2) / 2)

    coords = ["x", "y", "z"]
    for k in range(3):
        solver_k = exp_k2(solver, k).reshape(-1)
        model_k = exp_k2(model, k).reshape(-1)

        fig, ax = plt.subplots()

        ax.hist(solver_k, bins=50, density=True, alpha=0.6, label="Solver")
        ax.hist(model_k, bins=50, density=True, alpha=0.6, label="Model")

        ax.set_xlabel(f"exp(-{coords[k]}**2 / 2)")
        ax.set_yscale("log")
        ax.set_ylabel("density")
        ax.set_title(f"Histogram of {coords[k]}-coordinate statistic")
        ax.legend()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Analysis functions for autoregressive outputs",
    )
    parser.add_argument("dirname", help="directory containing npy files")
    parser.add_argument("--series", default=0, type=int, help="series to plot")
    parser.add_argument("--winsize", default=None, type=int, help="window size to plot")
    parser.add_argument(
        "--err_thresh",
        default=3,
        type=float,
        help="MSE threshold for distance accuracy score",
    )
    parser.add_argument(
        "--calc_score",
        action="store_true",
        default=False,
        help="calculate time to diverge",
    )
    parser.add_argument(
        "--compare_ivp",
        action="store_true",
        default=False,
        help="compare model, Radau, and RK45 outputs",
    )

    parser.add_argument(
        "--zmin",
        default=38.45,
        type=float,
        help="minimum z-coord value for z-based analysis",
    )
    parser.add_argument(
        "--zmax",
        default=38.6,
        type=float,
        help="maximum z-coord value for z-based analysis",
    )

    parser.add_argument(
        "--npts", default=None, type=int, help="number of points per series to plot"
    )
    parser.add_argument(
        "--minimal", action="store_true", default=False, help="minimal plot annotation"
    )
    args = parser.parse_args()

    md = np.load(f"{args.dirname}/md.npy", allow_pickle=True).item()
    ytrue = np.memmap(
        f"{args.dirname}/ytrue.npy", mode="r", dtype="float32", shape=md["shape"]
    )
    yhat = np.memmap(
        f"{args.dirname}/yhat.npy", mode="r", dtype="float32", shape=md["shape"]
    )

    spacing = getattr(md["config"], "spacing", 1)
    dt = spacing * md["dt"]

    print_minima(yhat, dt)

    if args.compare_ivp:
        compare_ivp(ytrue, yhat, dt)

    scores = []
    if args.calc_score:
        input_sec = dt * md["config"].input_size
        scores = calc_acc_dist(yhat, ytrue, args.err_thresh)
        print(
            f"average time to reach L2 diff={args.err_thresh}: {dt*scores.mean():.2f}s (input size: {input_sec:.2f}s)"
        )

    npts = args.npts
    if npts is None:
        npts = ytrue.shape[1]

    winsize = args.winsize
    if winsize is None:
        winsize = md["config"].H

    statistics(yhat, ytrue, dt)
    plot_compare_full(ytrue, yhat, 5, npts)
    z_traj = collect_z_traj(yhat, args.zmin, args.zmax)
    plot_3d(yhat, ytrue, z_traj, winsize, dt, args.minimal)

    plt.show()
    plt.close()
