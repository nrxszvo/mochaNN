import multiprocessing as mp
import os
from functools import partial

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from data_generation.generate_dataset import PERIOD, solve_lorenz

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

    gs = gridspec.GridSpec(ncomp, 2)
    gs.update(wspace=0, hspace=0)

    def plot(data, idx, color):
        ax = fig.add_subplot(gs[2 * i + idx], projection="3d")
        ax.plot(*data.T, color=color)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        if i == 0:
            ax.set_title("IVP Solver" if idx == 0 else "Model 3")

    for i, s in enumerate(series):
        plot(yt[s, :npts], 0, "C0")
        plot(yh[s, :npts], 1, "C1")
    fig.subplots_adjust(bottom=0, top=0.9)


def plot_3d(yh, yt, sidx, winsize, dt, scores):
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
    startidx = 0
    endidx = winsize

    ytwin = yt[sidx, startidx:endidx]
    yt3d = ax.plot(*ytwin.T, label="reference", alpha=1)[0]

    yhwin = yh[sidx, startidx:endidx]
    yh3d = ax.plot(*yhwin.T, label="prediction", alpha=1)[0]

    ax.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure)

    err = np.linalg.norm(ytwin - yhwin, axis=1).max()
    dfo = np.linalg.norm(ytwin, axis=1).min()

    def title(sidx, endidx, err, dfo):
        ax.set_title(
            f"""
                           series {sidx}

                   time = {dt*endidx:.2f},    max L2 diff = {err:.2f} dfo = {dfo:.2f}
            """,
            y=1,
            x=0.3,
        )

    title(sidx, endidx, err, dfo)
    # fig.subplots_adjust(left=-0.1, right=1.1, bottom=-0.1, top=1.1)

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

        if e.key in ["b", "f"]:
            if e.key == "b":
                startidx = max(0, startidx - winsize)
            else:
                startidx = min(endidx - winsize, startidx + winsize)
            update_plot(sidx, startidx, endidx)

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

        elif e.key == "p":
            sidx = (sidx + 1) % nseries
            endidx -= startidx
            startidx = 0
            update_plot(sidx, startidx, endidx)

        elif e.key == "q":
            sidx = (sidx + nseries - 1) % nseries
            startidx = npts - (endidx - startidx)
            endidx = npts
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
        "--npts", default=None, type=int, help="number of points per series to plot"
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

    plot_compare_full(ytrue, yhat, 5, npts)
    plot_3d(yhat, ytrue, args.series, winsize, dt, scores)

    plt.show()
    plt.close()
