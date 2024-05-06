import numpy as np


def collect_trajectories(solns, thresh, offset, N):
    nseries, npts, ndim = solns.shape
    dfo = np.linalg.norm(solns, axis=-1)
    traj = []
    candidates = np.argwhere(dfo < thresh)
    minima = []
    for s, i in candidates:
        if (
            i > 0
            and i < npts - 1
            and dfo[s, i - 1] > dfo[s, i]
            and dfo[s, i + 1] > dfo[s, i]
        ):
            minima.append((s, i))

    for s, i in minima:
        if i - offset > 0:
            start = max(0, i - offset - N)
            end = i - offset
            traj.append(solns[s, start:end])

    return traj


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Given npy(s) of solutions, select segments from the solutions that eventually pass within threshold distance of the origin, and save them to a file, to be used as initial conditions for generating new trajectories",
    )
    parser.add_argument("--inp", help="npy dataset or directory of datasets")
    parser.add_argument(
        "--thresh", default=1, type=float, help="L2 threshold for selecting trajectory"
    )
    parser.add_argument(
        "--end",
        default=1000,
        type=int,
        help="number of points before minimum where trajectory segment should end",
    )
    parser.add_argument(
        "--n", default=2000, type=int, help="length of trajectory segment to save"
    )
    parser.add_argument("--out", help="output npy file for holding trajectory segments")

    args = parser.parse_args()

    all_traj = []
    if os.path.isdir(args.inp):
        npys = [f"{args.inp}/{fn}" for fn in os.listdir(args.inp) if fn.endswith("npy")]
    else:
        npys = [args.inp]
    for npy in npys:
        print(f"{os.path.basename(npy)}: ", end="")
        d = np.load(npy, allow_pickle=True).item()
        traj = collect_trajectories(d["solutions"], args.thresh, args.end, args.n)
        del d
        print(len(traj))
        if len(traj) > 0:
            all_traj.append(np.stack(traj))

    np.save(args.out, np.array(all_traj), allow_pickle=True)
