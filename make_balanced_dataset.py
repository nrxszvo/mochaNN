import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Given an npy dataset file produced by 'generate_dataset.npy', 
        sort the solutions (series) according to their minimum L2-magnitude point,
        and construct a new dataset with train, val, and test sets having an equal 
        distribution of series based on their minimum L2 points.  Series are organized 
        such that all train series appear first, followed by validation series, followed by test series""",
    )
    parser.add_argument("in_npy", help="input dataset npy file")
    parser.add_argument("out_npy", help="output npy filename")
    parser.add_argument(
        "--n", type=int, default=10000, help="number of series to select"
    )
    parser.add_argument(
        "--n_val", default=100, type=int, help="number of series for validation"
    )
    parser.add_argument(
        "--n_test", default=200, type=int, help="number of series for test"
    )
    args = parser.parse_args()

    d = np.load(args.in_npy, allow_pickle=True).item()
    solns = d["solutions"]
    dfo = np.linalg.norm(solns, axis=2)
    _, npts = dfo.shape
    srs_sorted = np.argsort(dfo.min(axis=1))[: args.n]

    val_ivl = args.n // args.n_val
    test_ivl = args.n // args.n_test
    train_srs = []
    test_srs = []
    val_srs = []
    for i in range(args.n):
        s = srs_sorted[i]
        if i % test_ivl == 0:
            test_srs.append(s)
        elif (i - 1) % val_ivl == 0:
            val_srs.append(s)
        else:
            train_srs.append(s)

    series = np.concatenate([solns[train_srs], solns[val_srs], solns[test_srs]])
    d["solutions"] = series
    np.save(args.out_npy, d, allow_pickle=True)
