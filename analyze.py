from functools import partial
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams["keymap.back"].remove("left")
plt.rcParams["keymap.forward"].remove("right")
plt.rcParams["keymap.save"].remove("s")
plt.rcParams["keymap.pan"].remove("p")
plt.rcParams["keymap.zoom"].remove("o")
plt.rcParams["keymap.quit"].remove("q")
plt.rcParams["keymap.home"].remove("r")


def load(fn):
    return np.load(fn, allow_pickle=True).item()


def get_ys(d):
    yt = d["y_true"]
    yh = d["y_hat"]
    return yt, yh


def calc_mae(yt, yh, ax=(0, 1, 2, 3)):
    mae = np.abs(yt - yh)
    if len(ax) > 0:
        mae = mae.mean(axis=ax)
    return mae


def calc_smape(yt, yh, ax=(0, 1, 2, 3)):
    t = np.prod([yt.shape[i] for i in ax])
    smape = (200 / t) * np.sum(np.abs(yt - yh) / (np.abs(yt) + np.abs(yh)), axis=ax)
    return smape


def get_min_dist_errors(yt, smape):
    nseries, nwin, winsize, ndim = yt.shape
    yt_flat = yt[:, ::winsize].reshape(nseries, -1, ndim)
    dfo = np.linalg.norm(yt_flat, axis=2)
    npts = dfo.shape[1]
    ds = []
    es = []
    for s in range(nseries):
        s_ds = []
        for i in range(npts - 2):
            if dfo[s, i] - dfo[s, i + 1] > 0 and dfo[s, i + 1] - dfo[s, i + 2] < 0:
                s_ds.append((i + 1, dfo[s, i + 1]))
        for idx in range(len(s_ds)):
            i, dist = s_ds[idx]
            wstart = max(0, i - winsize + 1)
            max_e = 0
            for j in range(wstart, min(i + 1, nwin)):
                if smape[s, j] > max_e and dist <= dfo[s, j : j + winsize].min():
                    max_e = smape[s, j]

            ds.append(dist)
            es.append(max_e)

    return ds, es


def _plot_errors_by_window(d, name, fig, ax, smape_win=None):
    yt, yh = get_ys(d)
    nseries, nwin, winsize, ndim = yt.shape
    xax = np.arange(nwin)

    if smape_win is None:
        smape_win = calc_smape(yt, yh, ax=(2, 3))

    yt_flat = yt[:, ::winsize].reshape(nseries, -1, ndim)
    dist_from_origin = np.linalg.norm(yt_flat, axis=2)

    state = {"series": 0, "nseries": nseries, "name": name}

    fig.suptitle("average error by window")

    ls = ""
    m = "o"
    ms = 1
    smape_plt = ax.plot(
        xax,
        smape_win[state["series"]],
        label="sMAPE avg",
        linestyle=ls,
        marker=m,
        markersize=ms,
    )[0]

    dfo_plt = ax.twinx().plot(
        dist_from_origin[state["series"]], color="red", alpha=0.1
    )[0]

    state["smape_win"] = smape_win
    state["smape_plt"] = smape_plt
    state["smape_lim"] = (0.9 * smape_win.min(), 1.1 * smape_win.max())
    state["dfo"] = dist_from_origin
    state["dfo_plt"] = dfo_plt
    ax.set_ylim(*state["smape_lim"])
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

    def onclick(fig, ax, nwin, annot, d, name, state, e):
        if e.inaxes is ax:
            if e.dblclick:
                sidx = state["series"]
                widx = int(e.xdata + 0.5)
                annot.xy = (e.xdata, e.ydata)
                annot.set_text(f"{sidx}.{widx}")
                annot.set_visible(True)
                fig.canvas.draw_idle()
                plot_3d(d, name, sidx, widx)
                plt.show()
            else:
                annot.set_text("")
                annot.set_visible(False)
                fig.canvas.draw_idle()

    def onkeypress(fig, ax, state, e):
        if e.key in ["left", "right"]:
            series = state["series"]
            n_series = state["nseries"]
            if e.key == "left":
                series = (n_series + series - 1) % n_series
            else:
                series = (series + 1) % n_series
            state["series"] = series
            state["smape_plt"].set_ydata(state["smape_win"][series])
            state["dfo_plt"].set_ydata(state["dfo"][series])
            ax.set_title(f'{state["name"]} - Series {state["series"]}', fontsize=10)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    fig.canvas.mpl_connect(
        "button_press_event", partial(onclick, fig, ax, nwin, annot, d, name, state)
    )

    fig.canvas.mpl_connect("key_press_event", partial(onkeypress, fig, ax, state))

    return smape_win


def plot_dist(ds):
    figHist, axHist = plt.subplots()
    axHistT = axHist.twinx()
    axHIdx = plt.figure().add_subplot()
    axDFOIdx = plt.figure().add_subplot()
    figWErr, axesWErr = plt.subplots(len(ds), 1)

    if len(ds) == 1:
        axesWErr = [axesWErr]

    # axes are: series, window, widx, dim
    for i, ((d, name), axWErr) in enumerate(zip(ds, axesWErr)):
        smape_win = _plot_errors_by_window(d, name, figWErr, axWErr)
        if name == ds[-1][1]:
            axWErr.set_xlabel("window #")

        axHist3d = plt.figure().add_subplot(projection="3d")
        xs, ys = np.meshgrid(
            np.arange(smape_win.shape[0]), np.arange(smape_win.shape[1]), indexing="ij"
        )
        pts = np.stack([xs, ys, smape_win])
        for sidx in range(pts.shape[1]):
            axHist3d.plot(*pts[:, sidx])
        axHist3d.set_xlabel("series index")
        axHist3d.set_ylabel("window index")
        axHist3d.set_zlabel("sMAPE")
        axHist3d.set_title(f"{name} - error by series and window", y=1.0, pad=-14)
        axHist3d.get_figure().tight_layout()

        yt, yh = get_ys(d)

        err_hidx = calc_smape(yt, yh, ax=(0, 1, 3))
        axHIdx.scatter(np.arange(len(err_hidx)), err_hidx, s=0.5, label=name)

        dfos, errs = get_min_dist_errors(yt, smape_win)
        axDFOIdx.scatter(dfos, errs, s=5, label=name)

        axHist.hist(
            smape_win.reshape(-1), bins=100, density=True, alpha=0.6, label=name
        )
        vs, edges, patches = axHistT.hist(
            smape_win.reshape(-1),
            bins=100,
            cumulative=True,
            histtype="step",
            density=True,
            label=f"{name} CDF",
        )
        patches[0].set_xy(patches[0].get_xy()[:-1])
    axHIdx.set_xlabel("horizon index")
    axHIdx.set_ylabel("sMAPE")
    axHIdx.legend()
    axHIdx.set_title("Average error by horizon index")

    axDFOIdx.set_xlabel("distance from origin")
    axDFOIdx.set_ylabel("sMAPE")
    axDFOIdx.legend()
    axDFOIdx.set_title("Minimum distance from origin vs. maximum sMAPE")

    axHist.set_xlabel("sMAPE")
    axHist.set_ylabel("# windows")
    axHist.set_yscale("log")
    axHistT.set_ylabel("cumulative likelihood")
    figHist.legend(bbox_to_anchor=(0.88, 0.82))

    axHist.set_title(ds[-1][1] + " -- error distribution")
    plt.show()
    plt.close()


def plot_summary(ds):
    names = []
    errs = []
    for d, name in ds:
        names.append(name)
        yt, yh = get_ys(d)
        errs.append(calc_smape(yt, yh))
    plt.bar(names, errs)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_summary_menu(available):
    for i, (name, fn) in enumerate(available):
        print(f"{i}: {name}")
    inp = input("plot all or select? [a|s]: ")
    if inp == "a":
        plot_summary([(load(fn), name) for name, fn in available])
    else:
        choices_menu(available, plot_summary)


critical_points = np.array([[0, 0, 0], [8.49, 8.49, 27], [-8.49, -8.49, 27]])


def points_histo(d, name, start, end):
    yt = d["data"]
    fig, ax = plt.subplots()
    nseries, npts, ndim = yt.shape
    dfo = np.linalg.norm(yt[start:end], axis=2).reshape(-1)
    ax.hist(dfo, bins=200, alpha=0.6)
    axt = ax.twinx()
    vs, edges, patches = axt.hist(
        dfo,
        bins=200,
        cumulative=True,
        histtype="step",
        density=True,
        color="red",
        label="cdf",
    )
    patches[0].set_xy(patches[0].get_xy()[:-1])

    ax.set_xticks([0, 1, 2, 3, 5, 10, 20, 30, 40, 50])
    ax.set_xlabel("distance from origin")
    ax.set_ylabel("# points")
    ax.set_yscale("log")
    axt.set_yscale("log")
    axt.set_ylabel("cumulative distribution")
    # fig.legend(bbox_to_anchor=(1, 1))
    ax.set_title("Dataset points distribution")


def plot_3d_ref(d, name, sidx, eidx, pstart, pend):
    yt = d["data"]
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    cps = ax.scatter(*critical_points.T, label="critical points", color="purple")
    axObjs = []
    for i in range(sidx, eidx + 1):
        yti = yt[i, pstart:pend]
        ao = ax.plot(*yti.T)[0]
        axObjs.append(ao)
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure)
    fig.subplots_adjust(top=1.05, bottom=0.05)

    def update(yt, ax, state, frame):
        pstart = state["pstart"] + frame
        pend = state["pend"] + frame
        sidx = state["sidx"]
        eidx = state["eidx"]
        for s, ao in zip(range(sidx, eidx + 1), state["axObjs"]):
            yti = yt[s, pstart:pend]
            ao.set_data_3d(*yti.T)
        return state["axObjs"]

    def onkeypress(yt, fig, ax, state, e):
        pstart = state["pstart"]
        pend = state["pend"]
        nseries, winsize, ndim = yt.shape
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
                state["axObjs"][i].set_data_3d(*yt[sidx + i, pstart:pend].T)

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
                    yt,
                    ax,
                    state,
                ),
                frames=90,
                interval=100,
            )
            ani.save("trajectories.gif", dpi=50, writer=animation.PillowWriter(fps=10))
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
            yt,
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


def plot_3d(d, name, sidx, widx):
    yt, yh = get_ys(d)
    ytw = yt[sidx, widx]
    yhw = yh[sidx, widx]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*critical_points.T, label="critical points", color="purple")
    smape = calc_smape(ytw, yhw, ax=(0, 1))
    yt3d = ax.plot(*ytw.T, label="reference")[0]
    yh3d = ax.plot(*yhw.T, label="prediction", alpha=0.6)[0]
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure)

    ax.set_title(f"Window {widx} - sMAPE Error: {smape:.2f}", loc="left")
    # ax.set_title(f"Series {sidx}, Window {widx}, sMAPE Error: {smape:.2f}", loc="left")

    def concat_yt(yt, widx, nwinyt):
        inc = yt.shape[1] - 1
        start = widx - (inc * (nwinyt - 1))
        end = widx + inc
        return np.concatenate([yt[wi] for wi in np.arange(start, end, inc)])

    def update(yt, yh, ax, state, frame):
        widx = state["widx"] + frame
        sidx = state["sidx"]

        ytw = yt[sidx, widx]
        yhw = yh[sidx, widx]
        smape = calc_smape(ytw, yhw, ax=(0, 1))
        dfo = np.linalg.norm(ytw, axis=-1).min()
        ax.set_title(
            f"Window {widx} - sMAPE Error: {smape:.2f} - DFO {dfo:.1f}", loc="left"
        )
        state["yt3d"].set_data_3d(*ytw.T)
        state["yh3d"].set_data_3d(*yhw.T)
        return state["yt3d"], state["yh3d"]

    def onkeypress(yt, yh, fig, ax, state, e):
        sidx = state["sidx"]
        widx = state["widx"]
        nseries, nwin, winsize, ndim = yt.shape

        if e.key == "b":
            if widx >= winsize - 1:
                state["nwinyt"] += 1
                ytw = concat_yt(yt[sidx], widx, state["nwinyt"])
                state["yt3d"].set_data_3d(*ytw.T)
                fig.canvas.draw_idle()
        elif e.key in ["left", "right"]:
            if e.key == "left":
                if widx > 0:
                    widx -= 1
                else:
                    sidx = (nseries + sidx - 1) % nseries
                    widx = nwin - 1
            elif e.key == "right":
                if widx < nwin - 1:
                    widx += 1
                else:
                    sidx = (sidx + 1) % nseries
                    widx = 0

            ytw = concat_yt(yt[sidx], widx, state["nwinyt"])
            state["yt3d"].set_data_3d(*ytw.T)
            smape = calc_smape(yt[sidx, widx], yh[sidx, widx], ax=(0, 1))
            state["yh3d"].set_data_3d(*yh[sidx, widx].T)
            dfo = np.linalg.norm(ytw, axis=-1).min()
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
                    {
                        "sidx": sidx,
                        "widx": widx,
                        "yt3d": state["yt3d"],
                        "yh3d": state["yh3d"],
                    },
                ),
                frames=90,
                interval=100,
            )
            ani.save("animate.gif", dpi=50, writer=animation.PillowWriter(fps=10))

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
                "smape": smape,
                "nwinyt": 1,
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
    for fn in sorted(os.listdir(dirname)):
        m = re.match(pattern, fn)
        if m is not None:
            available.append((f"{dirname}/{fn}", m.group(1)))
    return available


def choices_menu(available, fn):
    for i, (name, _) in enumerate(available):
        print(f"{i}: {name}")
    choices = []
    while True:
        inp = input("idx to add (a for all): ")
        if inp == "a":
            fn([(load(fn), name) for fn, name in available])
        elif inp == "":
            if len(choices) > 0:
                fn([(load(fn), name) for fn, name in choices])
                choices = []
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


def plot_3d_menu(ds):
    if len(ds) > 1:
        raise Exception("plot_3d only supports one dataset")
    d, name = ds[0]
    try:
        nser = d["y_true"].shape[0]
        nwin = d["y_true"].shape[1]
        resp = input(f"{name}: series and window to plot [0, {nser-1}].[0, {nwin-1}]: ")
        sidx, widx = resp.split(".")
        plot_3d(d, name, int(sidx), int(widx))
    except Exception as e:
        print(e)
    plt.show()
    plt.close()


def print_metadata(ds):
    for d, name in ds:
        yt, yh = get_ys(d)
        mae = calc_mae(yt, yh)
        smape = calc_smape(yt, yh)
        print()
        print(name)
        if "dataset" in d:
            print(f'\tdataset: {d["dataset"]}')
        print(f"\tMAE: {mae:.3f}")
        print(f"\tsMAPE: {smape:.3f}")
        print("\tconfig:")
        for k, v in d["config"].items():
            print(f"\t\t{k}: {v}")
    print()


def plot_trajectories_menu(available):
    for i, (name, _) in enumerate(available):
        print(f"{i}: {name}")
    while True:
        inp = input("enter index: ")
        try:
            c = int(inp)
            fn = available[c][0]
            name = available[c][1]
            d = load(fn)
            nseries, winsize, ndim = d["data"].shape
            first = int(input(f"first series [0, {nseries-1}]: "))
            last = int(input(f"last series [{first}, {nseries-1}]: "))
            pstart = int(input(f"starting point [0, {winsize-1}]: "))
            pend = int(input(f"end point [{pstart+1}, {winsize}]: "))
            n = input("optional title for plot: ")
            if n != "":
                name = n
            plot_3d_ref(d, name, first, last, pstart, pend)
            plt.show()
            plt.close()
        except Exception as e:
            print(e)
            break


def plot_dataset_histo_menu(available):
    for i, (name, _) in enumerate(available):
        print(f"{i}: {name}")
    while True:
        inp = input("enter index: ")
        try:
            c = int(inp)
            fn = available[c][0]
            name = available[c][1]
            d = load(fn)
            nseries = d["data"].shape[0]
            try:
                start = int(input(f"start index [0, {nseries-1}]: "))
            except Exception:
                start = 0
            try:
                end = int(input(f"end index [{start+1},{nseries}]: "))
            except Exception:
                end = nseries
            points_histo(d, name, start, end)
            plt.savefig(f"{name}_points.png", dpi=500)
            plt.close()
        except Exception as e:
            print(e)
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--pattern", default="(.*).npy", help="re for matching npy files"
    )
    parser.add_argument(
        "--dirname", default="predictions", help="prediction data directory"
    )
    args = parser.parse_args()

    available = collect_available(args.pattern, args.dirname)

    while True:
        opt = input("enter dist|summary|3d|info|trajectories|points: ")

        if opt == "dist":
            choices_menu(available, plot_dist)
        elif opt == "3d":
            choices_menu(available, plot_3d_menu)
        elif opt == "summary":
            plot_summary_menu(available)
        elif opt == "info":
            choices_menu(available, print_metadata)
        elif opt == "trajectories":
            plot_trajectories_menu(available)
        elif opt == "points":
            plot_dataset_histo_menu(available)
        else:
            break
