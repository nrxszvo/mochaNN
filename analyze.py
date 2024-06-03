import os
from functools import partial

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from numpy.polynomial.polynomial import Polynomial

from data_generation.utils import get_local_minima, get_local_maxima

plt.rcParams["keymap.fullscreen"].remove("f")
plt.rcParams["keymap.back"].remove("left")
plt.rcParams["keymap.forward"].remove("right")
plt.rcParams["keymap.save"].remove("s")
plt.rcParams["keymap.pan"].remove("p")
plt.rcParams["keymap.zoom"].remove("o")
plt.rcParams["keymap.quit"].remove("q")
plt.rcParams["keymap.home"].remove("r")


def load_map(dn):
    fns = [fn for fn in os.listdir(dn) if fn.endswith("npy")]
    data = {}
    mdfn = list(filter(lambda fn: "md" in fn, fns))[0]
    d = np.load(f"{dn}/{mdfn}", allow_pickle=True).item()
    for k, v in d.items():
        data[k] = v

    for fn in fns:
        if len(data["shape"]) == 3:
            nseries, npts, ndim = data["shape"]
        else:
            nseries, nwin, winsize, ndim = data["shape"]
            npts = data["npts"]
        if "yhat" in fn:
            shape = (nseries, nwin, winsize, ndim)
            k = "y_hat"
        elif "ytrue" in fn:
            shape = (nseries, npts, ndim)
            k = "y_true"
        elif "solutions" in fn:
            shape = (nseries, npts, ndim)
            k = "solutions"
        else:
            continue
        d = np.memmap(f"{dn}/{fn}", dtype="float32", mode="r", shape=shape)
        data[k] = d

    return data


def get_ys(d):
    yt = d["y_true"]
    yh = d["y_hat"]
    return yt, yh


def yt_window(yt_flat, winsize, stride, ax=0):
    yt_tens = torch.from_numpy(yt_flat.copy())
    yt_win = yt_tens.unfold(ax, winsize, stride)
    dimidx = list(range(yt_win.ndim))
    dimidx[-1] = dimidx[-2]
    dimidx[-2] = len(dimidx) - 1
    return yt_win.permute(*dimidx).numpy()


def calc_mae(yt, yh, ax=(0, 1, 2, 3)):
    mae = np.abs(yt - yh)
    if len(ax) > 0:
        mae = mae.mean(axis=ax)
    return mae


def calc_smape(yt, yh, ax=(0, 1, 2, 3)):
    t = np.prod([yh.shape[i] for i in ax])
    smape = (200 / t) * np.sum(np.abs(yt - yh) / (np.abs(yt) + np.abs(yh)), axis=ax)
    return smape


def get_z_max_idx(y, idx):
    z = y[idx, 2]
    while z < y[idx - 1, 2]:
        idx -= 1
        z = y[idx, 2]
        if idx <= 0:
            raise Exception
    return idx


def get_min_dist_errors(yt, yh, stride):
    final_minima = []
    max_errs = []
    indices = []

    nseries, nwin, winsize, _ = yh.shape
    smapes = []
    z_maxes = {
        "z": [],
        "e": [],
        "si": [],
    }

    for s in tqdm.tqdm(range(nseries)):
        yt_win = yt_window(yt[s], winsize, stride)
        smape = calc_smape(yt_win, yh[s], ax=(1, 2))
        smapes.append(smape)
        dfo = np.linalg.norm(yt[s], axis=-1)
        minima, mindex = get_local_minima(dfo)
        for dist, i in zip(minima, mindex):
            # first find all windows that contain this local minima and for which it is the minimum L2 value in the window
            wstart = max(0, (i - winsize) // stride + 1)
            wend = min(i // stride + 1, nwin)
            min_dist_per_win = [
                dfo[w * stride : w * stride + winsize].min()
                for w in range(wstart, wend)
            ]
            idxs = np.argwhere(dist == min_dist_per_win)[:, 0]

            # then select the maximum window error among those windows
            if len(idxs) > 0:
                max_e = smape[wstart + idxs].max()
                max_i = smape[wstart + idxs].argmax()
                max_i = (wstart + idxs)[max_i]
                final_minima.append(dist)
                max_errs.append(max_e)
                indices.append((s, max_i))
                try:
                    zidx = get_z_max_idx(yt[s], i)
                    zm = yt[s, zidx, 2]
                    z_maxes["z"].append(zm)
                    z_maxes["e"].append(max_e)
                    z_maxes["si"].append((s, max_i))

                except Exception:
                    pass

    z_maxes["z"] = np.array(z_maxes["z"])
    z_maxes["e"] = np.array(z_maxes["e"])
    return (
        np.array(final_minima),
        np.array(max_errs),
        indices,
        np.array(smapes),
        z_maxes,
    )


def plot_dfo_vs_max_err(fig, ax, ds):
    # maximum errors for windows that include distance-from-origin local minima
    state = {}
    stateZ = {}
    all_smapes = {}
    ds = {name: d for d, name in ds}

    figZ, axZ = plt.subplots()
    axZ.set_title("max Z coordinate vs. max sMAPE")
    axZ.set_ylabel("sMAPE")
    axZ.set_xlabel("z coord")

    for name, d in ds.items():
        print(f"analyzing {name}...")
        yt = d["y_true"]
        yh = d["y_hat"]
        stride = d["stride"]
        nseries, nwin, winsize, ndim = yh.shape
        dfos, errs, indices, smapes, z_maxes = get_min_dist_errors(yt, yh, stride)
        ax.scatter(dfos, errs, s=5, label=name, alpha=0.4)

        ffit = Polynomial.fit(dfos, errs, 9)
        x = np.sort(dfos)
        ax.plot(x, ffit(x), label="best-fit", alpha=0.3)
        state[name] = (dfos, errs, indices)
        all_smapes[name] = smapes

        axZ.scatter(z_maxes["z"], z_maxes["e"], s=5, label=name, alpha=0.4)
        stateZ[name] = (z_maxes["z"], z_maxes["e"], z_maxes["si"])
    axZ.legend()

    # double click opens 3d plot of corresponding window
    def onclick(fig, ax, state, e):
        if e.inaxes is ax:
            if e.dblclick:
                dist = e.xdata
                err = e.ydata
                for name in state:
                    dfos, errs, indices = state[name]
                    selected = np.argmin((errs - err) ** 2 + (dfos - dist) ** 2)
                    sidx, widx = indices[selected]
                    plot_3d(ds[name], name, sidx, widx)
                plt.show()

    fig.canvas.mpl_connect(
        "button_press_event",
        partial(onclick, fig, ax, state),
    )
    figZ.canvas.mpl_connect("button_press_event", partial(onclick, figZ, axZ, stateZ))

    return all_smapes


def series_stats(yt, yh, winsize, stride, sidx):
    yt_win = yt_window(yt[sidx], winsize, stride)
    smape = calc_smape(yt_win, yh[sidx], ax=(1, 2))
    dfo = np.linalg.norm(yt_win, axis=-1).min(axis=-1)
    return smape, dfo


def plot_average_window_error_per_series(d, name, fig, ax):
    yt, yh = get_ys(d)
    nseries, nwin, winsize, ndim = yh.shape
    xax = np.arange(nwin)

    sidx = 0

    get_stats = partial(series_stats, yt, yh, winsize, d["stride"])

    smape_win, dfo = get_stats(sidx)

    state = {"series": sidx, "nseries": nseries, "name": name}

    fig.suptitle("average error by window")

    ls = ""
    m = "o"
    ms = 1
    smape_plt = ax.plot(
        xax,
        smape_win,
        label="sMAPE avg",
        linestyle=ls,
        marker=m,
        markersize=ms,
    )[0]

    axt = ax.twinx()
    dfo_plt = axt.plot(
        dfo,
        color="red",
        alpha=0.6,
        label="distance from origin",
    )[0]
    axt.grid()
    axt.set_ylim(0.125, 60)
    axt.set_yscale("log", base=2)
    axt.set_ylabel("distance from origin")

    state["yt"] = yt
    state["yh"] = yh
    state["smape_plt"] = smape_plt
    state["dfo_plt"] = dfo_plt
    ax.set_ylim(0, 180)
    ax.set_ylabel("error")
    ax.legend([smape_plt], [smape_plt.get_label()])
    ax.set_title(f'{state["name"]} - Series {state["series"]}', fontsize=10)

    ax.zorder = 1
    ax.patch.set_visible(False)

    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(0, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
    )

    # double click opens 3d plot of corresponding window
    def onclick(fig, ax, annot, d, name, state, e):
        if e.inaxes is ax:
            sidx = state["series"]
            widx = int(e.xdata + 0.5)
            annot.xy = (e.xdata, e.ydata)
            annot.set_text(f"{sidx}.{widx}")
            annot.set_visible(True)
            fig.canvas.draw_idle()
            if e.dblclick:
                plot_3d(d, name, sidx, widx)
                plt.show()
        else:
            annot.set_text("")
            annot.set_visible(False)
            fig.canvas.draw_idle()

    # arrows to navigate through series
    def onkeypress(fig, ax, state, e):
        annot.set_text("")
        annot.set_visible(False)
        fig.canvas.draw_idle()

        if e.key in ["left", "right"]:
            series = state["series"]
            n_series = state["nseries"]
            if e.key == "left":
                series = (n_series + series - 1) % n_series
            else:
                series = (series + 1) % n_series
            state["series"] = series
            smape_win, dfo = get_stats(series)
            state["smape_plt"].set_ydata(smape_win)
            state["dfo_plt"].set_ydata(dfo)
            ax.set_title(
                f'max smape: {smape_win.max():.2f} - {state["name"]} - Series {state["series"]}',
                fontsize=10,
            )
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    fig.canvas.mpl_connect(
        "button_press_event",
        partial(onclick, fig, ax, annot, d, name, state),
    )

    fig.canvas.mpl_connect("key_press_event", partial(onkeypress, fig, ax, state))


def plot_lorenz_map(ds):
    xs_trans = []
    ys_trans = []
    xs_no_trans = []
    ys_no_trans = []
    state = {}
    for d, name in ds:
        state[name] = {"indices": []}
        yt, yh = get_ys(d)
        idx_trans = []
        idx_no_trans = []
        for s in range(yt.shape[0]):
            pts, idx = get_local_maxima(yt[s, :, 2])
            repeats = np.argwhere(idx[1:] - idx[:-1] == 1)
            if repeats.shape[0] > 0:
                idx = np.delete(idx, repeats)
                pts = np.delete(pts, repeats)
            signs = np.sign(yt[s, idx, 0])
            indicators = np.convolve(signs, [1, -1], mode="valid")
            lobe_trans = (np.argwhere(indicators != 0))[:, 0]
            if lobe_trans[0] == 0:
                lobe_trans = lobe_trans[1:]
            no_trans = (np.argwhere(indicators == 0))[:, 0]
            if no_trans[0] == 0:
                no_trans = no_trans[1:]

            xs_trans.extend(pts[lobe_trans - 1])
            ys_trans.extend(pts[lobe_trans])
            xs_no_trans.extend(pts[no_trans - 1])
            ys_no_trans.extend(pts[no_trans])
            idx_trans.extend([(s, i) for i in idx[lobe_trans]])
            idx_no_trans.extend([(s, i) for i in idx[no_trans]])

        state[name]["data"] = (
            xs_trans,
            ys_trans,
            xs_no_trans,
            ys_no_trans,
            idx_trans,
            idx_no_trans,
            d["stride"],
        )

    fig, ax = plt.subplots()
    ax.scatter(xs_no_trans, ys_no_trans, s=0.2, label="no transition")
    ax.scatter(xs_trans, ys_trans, s=0.2, label="lobe transition")
    ax.set_title("Lorenz Map")
    ax.set_ylabel("Zmax(n+1)")
    ax.set_xlabel("Zmax(n)")
    ax.legend()

    # double click opens 3d plot of corresponding window
    def onclick(state, e):
        if e.inaxes is ax:
            if e.dblclick:
                zx = e.xdata
                zy = e.ydata
                for name in state:
                    (
                        xs_trans,
                        ys_trans,
                        xs_no_trans,
                        ys_no_trans,
                        idx_trans,
                        idx_no_trans,
                        stride,
                    ) = state[name]["data"]
                    err_trans = (zx - xs_trans) ** 2 + (zy - ys_trans) ** 2
                    err_no_trans = (zx - xs_no_trans) ** 2 + (zy - ys_no_trans) ** 2
                    if err_trans.min() < err_no_trans.min():
                        selected = np.argmin(err_trans)
                        sidx, widx = idx_trans[selected]
                    else:
                        selected = np.argmin(err_no_trans)
                        sidx, widx = idx_no_trans[selected]
                    widx = widx // stride
                    plot_3d(d, name, sidx, widx)
                plt.show()

    fig.canvas.mpl_connect(
        "button_press_event",
        partial(onclick, state),
    )
    plt.show()


def plot_dist(ds):
    figHist, axHist = plt.subplots()
    axHistT = axHist.twinx()
    figDFO, axDFO = plt.subplots()
    figWErr, axesWErr = plt.subplots(len(ds), 1)

    if len(ds) == 1:
        axesWErr = [axesWErr]

    # maximum errors for windows that include distance-from-origin local minima
    smapes = plot_dfo_vs_max_err(figDFO, axDFO, ds)

    for i, ((d, name), axWErr) in enumerate(zip(ds, axesWErr)):
        plot_average_window_error_per_series(d, name, figWErr, axWErr)
        if name == ds[-1][1]:
            axWErr.set_xlabel("window #")

        # 3d plot of window errors, series index X window index X smape error
        axHist3d = plt.figure().add_subplot(projection="3d")
        xs, ys = np.meshgrid(
            np.arange(smapes[name].shape[0]),
            np.arange(smapes[name].shape[1]),
            indexing="ij",
        )
        pts = np.stack([xs, ys, smapes[name]])
        for sidx in range(pts.shape[1]):
            axHist3d.plot(*pts[:, sidx])
        axHist3d.set_xlabel("series index")
        axHist3d.set_ylabel("window index")
        axHist3d.set_zlabel("sMAPE")
        axHist3d.set_title(f"{name} - error by series and window", y=1.0, pad=-14)
        axHist3d.get_figure().tight_layout()

        # histogram of average window errors
        axHist.hist(
            smapes[name].reshape(-1), bins=100, density=True, alpha=0.6, label=name
        )
        vs, edges, patches = axHistT.hist(
            smapes[name].reshape(-1),
            bins=100,
            cumulative=True,
            histtype="step",
            density=True,
            label=f"{name} CDF",
        )
        patches[0].set_xy(patches[0].get_xy()[:-1])

    axDFO.set_xlabel("distance from origin")
    axDFO.set_ylabel("sMAPE")
    axDFO.legend()
    axDFO.set_title("Minimum distance from origin vs. maximum sMAPE")
    axHist.set_xlabel("sMAPE")
    axHist.set_ylabel("density")
    axHist.set_yscale("log")
    axHistT.set_ylabel("cumulative likelihood")
    figHist.legend(bbox_to_anchor=(0.88, 0.82))
    plt.show()
    plt.close()


CRITICAL_POINTS = np.array([[0, 0, 0], [8.49, 8.49, 27], [-8.49, -8.49, 27]])


def plot_3d_ref(solns, name, sidx, eidx, pstart, pend):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.set_xlim3d(left=-20, right=20)
    ax.set_ylim3d(bottom=-20, top=20)
    ax.set_zlim3d(bottom=0, top=50)

    cps = ax.scatter(*CRITICAL_POINTS.T, label="critical points", color="purple")
    axObjs = []
    for i in range(sidx, eidx + 1):
        yti = solns[i, pstart:pend]
        ao = ax.plot(*yti.T)[0]
        axObjs.append(ao)
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure)
    fig.subplots_adjust(top=1.05, bottom=0.05)

    def update(solns, ax, state, frame):
        pstart = state["pstart"] + frame
        pend = state["pend"] + frame
        sidx = state["sidx"]
        eidx = state["eidx"]
        for s, ao in zip(range(sidx, eidx + 1), state["axObjs"]):
            yti = solns[s, pstart:pend]
            ao.set_data_3d(*yti.T)
        return state["axObjs"]

    def onkeypress(solns, fig, ax, state, e):
        pstart = state["pstart"]
        pend = state["pend"]
        nseries, winsize, ndim = solns.shape
        sidx = state["sidx"]
        eidx = state["eidx"]
        if e.key in ["left", "right"]:
            if e.key == "left":
                if pstart > 0:
                    pstart -= 1
                    pend -= 1

            elif e.key == "right":
                if pend < winsize - 1:
                    pstart += 1
                    pend += 1

            for i in range(eidx + 1 - sidx):
                state["axObjs"][i].set_data_3d(*solns[sidx + i, pstart:pend].T)

            state["pstart"] = pstart
            state["pend"] = pend
            fig.canvas.draw_idle()
        elif e.key == "r":
            ax.legend(
                loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure
            )
            ani = animation.FuncAnimation(
                fig,
                func=partial(
                    update,
                    solns,
                    ax,
                    state,
                ),
                frames=FRAMES,
                interval=INTERVAL,
            )
            ani.save(
                "trajectories.gif", dpi=DPI, writer=animation.PillowWriter(fps=FPS)
            )
        elif e.key == "p":
            vis = state["cps"].get_visible()
            state["cps"].set_visible(not vis)
            if vis:
                ax.legend().remove()
            else:
                ax.legend(
                    loc="upper right",
                    bbox_to_anchor=(1, 1),
                    bbox_transform=fig.transFigure,
                )
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect(
        "key_press_event",
        partial(
            onkeypress,
            solns,
            fig,
            ax,
            {
                "pstart": pstart,
                "pend": pend,
                "sidx": sidx,
                "eidx": eidx,
                "axObjs": axObjs,
                "cps": cps,
            },
        ),
    )


def concat_y(y, widx, ncat, stride):
    idx_per_win = y.shape[1] * stride
    start = widx
    end = widx + ncat * idx_per_win
    return np.concatenate([y[wi] for wi in np.arange(start, end, idx_per_win)])


def concat_rev(y, widx, ncat, stride):
    nwin, winsize, ndim = y.shape
    idx_per_win = winsize // stride

    if widx < idx_per_win:
        return y[0, : stride * widx]

    segment = np.empty((0, ndim))

    start = widx - (idx_per_win * ncat)
    if start < 0:
        segment = y[0, : stride * (widx % idx_per_win)]
        start = widx % idx_per_win

    segment = np.concatenate(
        [segment] + [y[wi] for wi in np.arange(start, widx, idx_per_win)],
    )

    return segment


def plot_3d(d, name, sidx, widx):
    yt, yh = get_ys(d)

    stride = d["stride"] if "stride" in d else 1
    winsize = yh.shape[2]

    ytw = concat_y(yt_window(yt[sidx], winsize, stride), widx, NCAT, stride)
    yhw = concat_y(yh[sidx], widx, NCAT, stride)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(*CRITICAL_POINTS.T, label="critical points", color="purple")
    if ytw.shape[0] == 1:
        yt3d = ax.scatter(*ytw.T, label="reference")
        yh3d = ax.scatter(*yhw.T, label="prediction", alpha=0.6)
    else:
        yt3d = ax.plot(*ytw.T, label="reference")[0]
        yh3d = ax.plot(*yhw.T, label="prediction", alpha=0.6)[0]
    ax.legend(
        loc="upper right", bbox_to_anchor=(1, 0.9), bbox_transform=fig.transFigure
    )

    ax.set_xlim3d(left=-20, right=20)
    ax.set_xlabel("x")
    ax.set_ylim3d(bottom=-20, top=20)
    ax.set_ylabel("y")
    ax.set_zlim3d(bottom=0, top=50)
    ax.set_zlabel("z")

    dfo = np.linalg.norm(ytw, axis=-1).min()
    smape = calc_smape(ytw, yhw, ax=(0, 1))

    ax.set_title(
        f"{name} - Series {sidx} - Window {widx} - sMAPE Error: {smape:.2f} - DFO {dfo:.1f}",
        loc="left",
    )

    def update(yt, yh, ax, state, frame):
        widx = state["widx"] + INC * frame
        sidx = state["sidx"]
        winsize = yh.shape[2]
        yt_fold = yt_window(yt[sidx], winsize, stride)
        ytw = concat_y(yt_fold, widx, NCAT, stride)
        yhw = concat_y(yh[sidx], widx, NCAT, stride)

        smape = calc_smape(yt[sidx, widx], yh[sidx, widx], ax=(0, 1))
        dfo = np.linalg.norm(ytw, axis=-1).min()
        ax.set_title(f"sMAPE Error: {smape:.2f} - DFO {dfo:.1f}", loc="left")
        state["yt3d"].set_data_3d(*ytw.T)
        state["yh3d"].set_data_3d(*yhw.T)
        ret = (state["yt3d"], state["yh3d"])
        if "yinput3d" in state:
            ywin = concat_rev(yt_fold, widx, state["nwininput"], stride)
            state["yinput3d"].set_data_3d(*ywin.T)
            ret = (state["yt3d"], state["yh3d"], state["yinput3d"])
        return ret

    def onkeypress(yt, yh, fig, ax, state, e):
        sidx = state["sidx"]
        widx = state["widx"]
        inc = state["inc"]
        nseries, nwin, winsize, ndim = yh.shape

        if e.key in ["b", "f"]:
            if e.key == "b":
                state["nwininput"] += 1
            elif state["nwininput"] > 0:
                state["nwininput"] -= 1
            if state["nwininput"] > 0:
                ywin = concat_rev(
                    yt_window(yt[sidx], winsize, stride),
                    widx,
                    state["nwininput"],
                    stride,
                )
                if "yinput3d" in state:
                    state["yinput3d"].set_data_3d(*ywin.T)
                else:
                    state["yinput3d"] = ax.plot(*ywin.T, label="input", alpha=0.6)[0]
            elif "yinput3d" in state:
                state["yinput3d"].remove()
                del state["yinput3d"]
            fig.canvas.draw_idle()

        elif any([k in e.key for k in ["left", "right"]]):
            if "shift" in e.key:
                inc *= 10
            if "left" in e.key:
                if widx >= inc:
                    widx -= inc
                else:
                    sidx = (nseries + sidx - 1) % nseries
                    widx = nwin - 1
            elif "right" in e.key:
                if widx < nwin - inc:
                    widx += inc
                else:
                    sidx = (sidx + 1) % nseries
                    widx = 0

            yt_fold = yt_window(yt[sidx], winsize, stride)
            ytw = concat_y(yt_fold, widx, state["ncat"], stride)
            yhw = concat_y(yh[sidx], widx, state["ncat"], stride)
            if ytw.shape[0] == 1:
                state["yt3d"].remove()
                state["yt3d"] = ax.scatter(*ytw.T, label="reference", color="blue")
                state["yh3d"].remove()
                state["yh3d"] = ax.scatter(*yhw.T, label="prediction", color="orange")
            else:
                state["yt3d"].set_data_3d(*ytw.T)
                state["yh3d"].set_data_3d(*yhw.T)
                if "yinput3d" in state:
                    ywin = concat_rev(yt_fold, widx, state["nwininput"], stride)
                    state["yinput3d"].set_data_3d(*ywin.T)

            dfo = np.linalg.norm(ytw, axis=-1).min()
            npts = yhw.shape[0]
            smape = calc_smape(ytw[-npts:], yhw, ax=(0, 1))
            ax.set_title(
                f"Window {widx} - sMAPE Error: {smape:.2f} - DFO {dfo:.3f}", loc="left"
            )

            state["widx"] = widx
            state["sidx"] = sidx
            fig.canvas.draw_idle()

        elif e.key == "r":
            ax.legend(
                loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure
            )
            ani = animation.FuncAnimation(
                fig,
                func=partial(
                    update,
                    yt,
                    yh,
                    ax,
                    state,
                ),
                frames=min(FRAMES, nwin - widx),
                interval=INTERVAL,
            )
            ani.save("animate.gif", dpi=100, writer=animation.PillowWriter(fps=FPS))

    fig.canvas.mpl_connect(
        "key_press_event",
        partial(
            onkeypress,
            yt,
            yh,
            fig,
            ax,
            {
                "sidx": sidx,
                "widx": widx,
                "yt3d": yt3d,
                "yh3d": yh3d,
                "inc": INC,
                "ncat": NCAT,
                "nwininput": 0,
            },
        ),
    )


def print_hparams(d):
    print(d["series"][0])
    print(f'\tlearning rate: {d["config"]["learning_rate"]}')
    print(f'\tkernel size: {d["config"]["n_pool_kernel_size"]}')
    print(f'\tdownsample: {d["config"]["n_freq_downsample"]}')
    print(f'\tmlp units: {d["config"]["mlp_units"]}')


def collect_available(pattern, dirname):
    available = []
    for curdir, dns, fns in os.walk(dirname):
        available.extend([(f"{curdir}/{dn}", dn) for dn in dns])
    return sorted(available, key=lambda a: a[0])


def choices_menu(available, func, load_func):
    def print_opts():
        print()
        for i, (fn, name) in enumerate(available):
            print(f"{i}: {name}")
        print()

    print_opts()
    choices = []
    while True:
        inp = input("idx to add (a for all): ")
        if inp == "a":
            func([(load_func(fn), name) for fn, name in available])
        elif inp == "":
            if len(choices) > 0:
                func([(load_func(fn), name) for fn, name in choices])
                choices = []
                print_opts()
            else:
                break
        else:
            try:
                c = int(inp)
                name = available[c][1]
                n = input("optional title for plot: ")
                if n != "":
                    name = n
                choices.append((available[c][0], name))
            except Exception as e:
                print(e)


def print_metadata(ds):
    for d, name in ds:
        yt, yh = get_ys(d)
        if yt.ndim == 3:
            nseries, nwin, winsize, ndim = yh.shape
            stride = d["stride"]
            maes = []
            smapes = []
            for sidx in tqdm.tqdm(range(nseries)):
                yt_win = yt_window(yt[sidx], winsize, stride)
                maes.append(calc_mae(yt_win, yh[sidx], ax=(0, 1, 2)))
                smapes.append(calc_smape(yt_win, yh[sidx], ax=(0, 1, 2)))
            mae = np.mean(maes)
            smape = np.mean(smapes)
        else:
            mae = calc_mae(yt, yh)
            smape = calc_smape(yt, yh)

        print()
        print(name)
        if "dataset" in d:
            print(f'\tdataset: {d["dataset"]}')
        print(f"\tMAE: {mae:.4f}")
        print(f"\tsMAPE: {smape:.4f}")
        print("\tconfig:")
        cfg = d["config"]
        if not isinstance(cfg, dict):
            cfg = cfg.__dict__
        for k, v in cfg.items():
            if isinstance(v, dict):
                print(f"\t\t{k}:")
                for vk, vv in v.items():
                    print(f"\t\t\t{vk}: {vv}")
            else:
                print(f"\t\t{k}: {v}")
    input("\n<enter> to continue")


def plot_trajectories_menu(available, load_func):
    for i, (name, _) in enumerate(available):
        print(f"{i}: {name}")
    while True:
        inp = input("enter index: ")
        try:
            c = int(inp)
            fn = available[c][0]
            name = available[c][1]
            d = load_func(fn)
            k = "solutions" if "solutions" in d else "data"
            nseries, winsize, ndim = d[k].shape
            first = int(input(f"first series [0, {nseries-1}]: "))
            last = int(input(f"last series [{first}, {nseries-1}]: "))
            pstart = int(input(f"starting point [0, {winsize-1}]: "))
            pend = int(input(f"end point [{pstart+1}, {winsize}]: "))
            n = input("optional title for plot: ")
            if n != "":
                name = n
            plot_3d_ref(d[k], name, first, last, pstart, pend)
            plt.show()
            plt.close()
        except Exception as e:
            print(e)
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="A collection of functions for analyzing and visualizing trajectory data from npy files, either datasets or prediction files",
    )
    parser.add_argument(
        "--pattern", default="(.*).npy", help="re for matching npy files"
    )
    parser.add_argument(
        "--dirname", default="predictions/maps", help="prediction data directory"
    )
    parser.add_argument(
        "--ncat",
        default=1,
        type=int,
        help="number of windows to concatenate together for 3d plot",
    )
    parser.add_argument(
        "--inc",
        default=1,
        type=int,
        help="number of points per frame to increment by when animating trajectories",
    )
    parser.add_argument(
        "--frames", default=500, help="number of frames to record", type=int
    )
    parser.add_argument("--fps", default=50, help="fps for gif", type=int)
    parser.add_argument("--dpi", default=100, help="dpi for gif/imaage", type=int)
    args = parser.parse_args()

    NCAT = args.ncat
    INC = args.inc
    FRAMES = args.frames
    FPS = args.fps
    INTERVAL = 1000.0 / FPS
    DPI = args.dpi
    available = collect_available(args.pattern, args.dirname)

    while True:
        opt = input("enter dist|info|trajectories|lorenz_map: ")

        if opt == "dist":
            choices_menu(available, plot_dist, load_map)
        elif opt == "info":
            choices_menu(available, print_metadata, load_map)
        elif opt == "trajectories":
            plot_trajectories_menu(available, load_map)
        elif opt == "lorenz_map":
            choices_menu(available, plot_lorenz_map, load_map)
        else:
            break
