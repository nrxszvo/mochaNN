import numpy as np
from scipy.integrate import solve_ivp
import multiprocessing as mp
import os
from collections import defaultdict
from functools import partial
import tqdm
import time
from utils import get_local_minima_from_solutions


def print_local_minima(solns, bins=[5, 4, 3, 2, 1]):
    minima, _ = get_local_minima_from_solutions(solns)
    hist = defaultdict(lambda: 0)
    for bd in bins:
        n = (minima < bd).sum()
        hist[bd] += n

    print("number of local minima with L2 less than:")
    for bd in bins:
        print(f"\t{bd}: {hist[bd]}")


def Lorenz(t, X):
    beta = 2.667
    rho = 28
    sigma = 10
    x, y, z = X
    xdot = sigma * y - sigma * x
    ydot = rho * x - x * z - y
    zdot = x * y - beta * z
    return xdot, ydot, zdot


NDIM = 3
PERIOD = 1.5008
DT = 0.0001801


def func(N, rpoints, ic):
    tlim = PERIOD * N / rpoints
    t_eval = np.linspace(0, tlim - tlim / N, N)
    R = solve_ivp(
        Lorenz,
        (t_eval[0], t_eval[-1]),
        ic,
        t_eval=t_eval,
        first_step=DT,
        method="Radau",
        vectorized=True,
    )
    if R.status != 0:
        raise Exception(f"solve_ivp error: {R.message}")
    return R.y.T


def ic_from_trajectory(traj, n_ic, seqlen, perturb, ndim):
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
    fn_func,
):
    pbf = perturb_factors(ic_perturb, n_ic, NDIM)

    if ic_method == "trajectory":
        ics = ic_from_trajectory(traj, n_ic, seqlen, pbf, NDIM)
    elif ic_method == "point":
        ics = ic_from_point(pbf, ic_point)

    n_proc = os.cpu_count()
    chunksize = 1
    sol_per_block = n_ic // blocks
    total = 0
    cur_block = 0
    tgt = partial(func, seqlen, resample_points)
    with mp.Pool(processes=n_proc) as pool:
        while total < n_ic:
            print(f"{n_ic - total} remaining")
            solns = []
            n_ic_block = min(n_ic - total, sol_per_block)
            cur_ics = ics[cur_block * n_ic_block : (cur_block + 1) * n_ic_block]
            for sol in tqdm.tqdm(
                pool.imap_unordered(
                    tgt,
                    cur_ics,
                    chunksize=chunksize,
                ),
                total=n_ic_block,
            ):
                solns.append(sol)

            total += len(solns)

            solns = np.array(solns)
            dataset = {
                "model": "Lorenz",
                "ndim": NDIM,
                "dt": PERIOD / resample_points,
                "dt_solver": DT,
                "solutions": solns,
            }
            np.save(fn_func(cur_block), dataset, allow_pickle=True)
            cur_block += 1

            start = time.time()
            print_local_minima(solns)
            end = time.time()
            print(f"time to compute local minima: {end-start:.1e} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate a set of solutions to the Lorenz Attractor using scipy.integrate.solve_ivp for a given number of initial conditions.  Calls to solve_ivp are parallelized using multiprocessing.",
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

    if args.blocks > 1:
        outdir = os.path.splitext(os.path.basename(args.fn))[0]
        os.makedirs(outdir, exist_ok=True)
        fn_func = lambda blk: f"{outdir}/blk-{blk}.npy"
    else:
        fn_func = lambda blk: args.fn

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
        fn_func,
    )
