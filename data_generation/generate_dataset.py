import numpy as np
from scipy.integrate import solve_ivp
import multiprocessing as mp
import os
from collections import defaultdict
from functools import partial
import tqdm
import time
import datetime
from .utils import get_local_minima_from_solutions


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


def solve_lorenz(N, rpoints, method, ic):
    tlim = PERIOD * N / rpoints
    t_eval = np.linspace(0, tlim - tlim / N, N)
    R = solve_ivp(
        Lorenz,
        (t_eval[0], t_eval[-1]),
        ic,
        t_eval=t_eval,
        first_step=DT,
        method=method,
    )
    if R.status != 0:
        raise Exception(f"solve_ivp error: {R.message}")
    return R.y.T


def perturb_factors(percent, n_ic, ndim):
    return 1 + percent - (2 * percent) * np.random.random((n_ic, ndim))


def generate_ics(n_ic, seqlen, ic_perturb, ic_points):
    pbf = perturb_factors(ic_perturb, n_ic, NDIM)
    ic_idx = np.random.choice(len(ic_points), n_ic)
    ics = ic_points[ic_idx] * pbf
    return ics


def make_multi_ic(
    n_ic,
    seqlen,
    resample_points,
    generate_ics,
    thresh,
    offset,
    outdir,
):
    shape = (n_ic, seqlen, NDIM)
    md = {
        "model": "Lorenz",
        "ndim": NDIM,
        "dt": PERIOD / resample_points,
        "dt_solver": DT,
        "shape": shape,
    }
    np.save(f"{outdir}/md.npy", md, allow_pickle=True)

    output = np.memmap(
        f"{outdir}/solutions.npy",
        dtype="float32",
        mode="w+",
        shape=shape,
    )

    n_proc = os.cpu_count()
    tgt = partial(solve_lorenz, seqlen, resample_points, "Radau")
    chunksize = 1

    blocksize = min(n_ic, 10000)

    total = 0
    ic_idx = 0
    ics = generate_ics()

    start = time.time()
    eta = "tbd"

    with mp.Pool(processes=n_proc) as pool:
        while total < n_ic:
            print(f"{n_ic - total} remaining (eta: {eta})")

            solns = []
            n_ic_block = min(n_ic - total, blocksize)

            if ic_idx + n_ic_block > len(ics):
                ics = generate_ics()
                ic_idx = 0

            cur_ics = ics[ic_idx : ic_idx + n_ic_block]
            ic_idx += n_ic_block

            for sol in tqdm.tqdm(
                pool.imap_unordered(
                    tgt,
                    cur_ics,
                    chunksize=chunksize,
                ),
                total=n_ic_block,
            ):
                if np.linalg.norm(sol[offset:], axis=-1).min() < thresh:
                    solns.append(sol)

            nsol = len(solns)
            if nsol > 0:
                solns = np.array(solns)
                output[total : total + nsol] = solns
                total += nsol
                print_local_minima(solns)

            if total > 0:
                end = time.time()
                eta = str(
                    datetime.timedelta(seconds=(n_ic - total) * (end - start) / total)
                )


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
        "--ic_points",
        help="npy containing points to use for initial condition selection",
        default=None,
    )
    parser.add_argument(
        "--thresh",
        default=5,
        type=float,
        help="require all trajectories to have a minimum L2 less than thresh",
    )
    parser.add_argument(
        "--offset",
        help="offset into solution from which to begin search for L2 minimum",
        default=200,
        type=int,
    )
    parser.add_argument("--outdir", default=None, help="output directory name")
    args = parser.parse_args()

    ic_points = np.load(args.ic_points)
    ic_points = ic_points.reshape(-1, NDIM)
    ic_func = partial(
        generate_ics,
        args.num_ic,
        args.seqlen,
        args.ic_perturb,
        ic_points,
    )

    os.makedirs(args.outdir, exist_ok=True)
    make_multi_ic(
        args.num_ic,
        args.seqlen,
        args.resample_points,
        ic_func,
        args.thresh,
        args.offset,
        args.outdir,
    )
