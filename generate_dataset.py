import numpy as np
import dysts.flows as flows
import multiprocessing as mp
import os
from collections import defaultdict
from functools import partial
import tqdm


def print_local_minima(solns):
    dfo = np.linalg.norm(solns, axis=2)
    hist = defaultdict(lambda: 0)
    for s in range(dfo.shape[0]):
        lm = []
        for p in range(1, dfo.shape[1] - 1):
            if dfo[s, p - 1] > dfo[s, p] and dfo[s, p + 1] > dfo[s, p]:
                lm.append(dfo[s, p])
        lm = np.array(lm)
        for bd in [5, 4, 3, 2, 1]:
            n = (lm < bd).sum()
            hist[bd] += n
    print("number of local minima with L2 less than:")
    for k in hist:
        print(f"\t{k}: {hist[k]}")


def func(model_fn, seqlen, resample_points, ic):
    model = model_fn()
    model.ic = ic
    sol = model.make_trajectory(n=seqlen, pts_per_period=resample_points)
    return sol


def ic_from_trajectory(model_fn, n_ic, seqlen, perturb, ndim):
    model = model_fn()
    sol_ref = model.make_trajectory(n=seqlen)
    ic_idx = np.random.choice(seqlen, n_ic, replace=False)
    ics = sol_ref[ic_idx] * perturb
    return ics


def ic_from_point(perturb, x0):
    return perturb * x0


def perturb_factors(percent, n_ic, ndim):
    return 1 + percent - (2 * percent) * np.random.random((n_ic, ndim))


def make_multi_ic(
    name,
    n_ic,
    seqlen,
    ic_perturb,
    resample_points,
    ic_method,
    ic_point,
):
    model_fn = getattr(flows, name)
    model = model_fn()
    model_md = model._load_data()
    ndim = model_md["embedding_dimension"]

    pbf = perturb_factors(ic_perturb, n_ic, ndim)

    if ic_method == "trajectory":
        ics = ic_from_trajectory(model_fn, n_ic, seqlen, pbf, ndim)
    elif ic_method == "point":
        if ic_point is None:
            ic_point = model.ic
        ics = ic_from_point(pbf, ic_point)

    n_proc = os.cpu_count()
    pool = mp.Pool(n_proc)
    solns = []
    chunksize = 10
    for sol in tqdm.tqdm(
        pool.imap_unordered(
            partial(func, model_fn, seqlen, resample_points), ics, chunksize=chunksize
        ),
        total=n_ic,
    ):
        solns.append(sol)

    pool.close()

    solns = np.array(solns)
    print_local_minima(solns)

    dataset = {
        "model": name,
        "ndim": ndim,
        "dt": model_md["period"] / resample_points,
        "dt_solver": model.dt,
        "lyapunov": model_md["maximum_lyapunov_estimated"],
        "period": model_md["period"],
        "solutions": solns,
    }
    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Use the 'dysts' module to generate a set of solutions for a given flow model and number of initial conditions.  Solutions are generated in parallel using multiprocessing.",
    )
    parser.add_argument(
        "--model", default="Lorenz", help="model name", choices=["Lorenz"]
    )
    parser.add_argument("--seqlen", default=10000, help="sequence length", type=int)
    parser.add_argument(
        "--ic_perturb",
        default=0.01,
        help="percent random perturbation of initial conditions",
        type=float,
    )
    parser.add_argument(
        "--num_ic",
        default=100,
        help="number of unique initial conditions to generate",
        type=int,
    )
    parser.add_argument(
        "--resample_points",
        default=100,
        type=int,
        help="resample points per period",
    )
    parser.add_argument(
        "--ic_method",
        default="point",
        help="method for generation of random initial conditions",
        choices=["trajectory", "point"],
    )
    parser.add_argument(
        "--ic_point",
        default=None,
        type=float,
        nargs="+",
        help="N-dimensional point to use with 'point' method for ic generation; if None, then the model's default ic will be used",
    )
    parser.add_argument("--fn", default=None, help="output filename")
    args = parser.parse_args()

    dataset = make_multi_ic(
        args.model,
        args.num_ic,
        args.seqlen,
        args.ic_perturb,
        args.resample_points,
        args.ic_method,
        args.ic_point,
    )
    np.save(args.fn, dataset, allow_pickle=True)
