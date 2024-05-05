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


def ic_from_trajectory(model_fn, traj, n_ic, seqlen, perturb, ndim):
    model = model_fn()
    if traj is None:
        traj = model.make_trajectory(n=seqlen)
    ic_idx = np.random.choice(len(traj), n_ic)
    ics = traj[ic_idx] * perturb
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
    traj,
    blocks,
):
    model_fn = getattr(flows, name)
    model = model_fn()
    model_md = model._load_data()
    ndim = model_md["embedding_dimension"]

    pbf = perturb_factors(ic_perturb, n_ic, ndim)

    if ic_method == "trajectory":
        ics = ic_from_trajectory(model_fn, traj, n_ic, seqlen, pbf, ndim)
    elif ic_method == "point":
        if ic_point is None:
            ic_point = model.ic
        ics = ic_from_point(pbf, ic_point)

    n_proc = os.cpu_count()
    chunksize = 1
    sol_per_block = n_ic // blocks
    total = 0
    cur_block = 0
    while total < n_ic:
        print(f"total: {total}")
        pool = mp.Pool(n_proc)
        solns = []
        n_ic_block = min(n_ic - total, sol_per_block)
        cur_ics = ics[cur_block * n_ic_block : (cur_block + 1) * n_ic_block]
        for sol in tqdm.tqdm(
            pool.imap_unordered(
                partial(func, model_fn, seqlen, resample_points),
                cur_ics,
                chunksize=chunksize,
            ),
            total=n_ic_block,
        ):
            solns.append(sol)

        total += len(solns)
        pool.close()
        solns = np.array(solns)
        print_local_minima(solns)

        dataset = {
            "model": "Lorenz",
            "ndim": ndim,
            "dt": model_md["period"] / resample_points,
            "dt_solver": model.dt,
            "solutions": solns,
        }

        np.save(f"blk-{cur_block}_{args.fn}", dataset, allow_pickle=True)
        cur_block += 1


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
    parser.add_argument(
        "--traj",
        help="trajectories file for use with ic_method==trajectory",
        default=None,
    )
    parser.add_argument("--fn", default=None, help="output filename")
    parser.add_argument(
        "--blocks",
        default=1,
        type=int,
        help="number of equal-sized npy files to divide the output among",
    )
    args = parser.parse_args()

    traj = args.traj
    if args.traj is not None:
        traj = np.load(args.traj)
        nseries, npts, ndim = traj.shape
        traj = traj.reshape(-1, ndim)

    make_multi_ic(
        args.model,
        args.num_ic,
        args.seqlen,
        args.ic_perturb,
        args.resample_points,
        args.ic_method,
        args.ic_point,
        traj,
        args.blocks,
    )
