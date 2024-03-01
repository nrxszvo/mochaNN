import numpy as np
import dysts.flows as flows


def make_multi_ic(name, n_ic, seqlen, ic_perturb, fn=None):
    if fn is None:
        fn = f"{name}_{n_ic}x{seqlen}_ic-{ic_perturb}"
    model = getattr(flows, name)()
    model_md = model._load_data()
    ndim = model_md["embedding_dimension"]
    perturb = 1 + ic_perturb - (2 * ic_perturb) * np.random.random((n_ic, ndim))
    perturb[0] = np.ones(ndim)  # first ic is default
    model.ic = model.ic[None, :] * perturb
    tpts, sol = model.make_trajectory(n=seqlen, return_times=True)

    dataset = {
        "model": name,
        "ndim": ndim,
        "dt": tpts[1],
        "lyapunov_time": 1 / model_md["maximum_lyapunov_estimated"],
        "period": model_md["period"],
        "ic": model.ic.tolist(),
        "solutions": sol,
    }
    np.save(f"{fn}.npy", dataset)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
    parser.add_argument("--fn", default=None, help="output filename")
    args = parser.parse_args()
    make_multi_ic(args.model, args.num_ic, args.seqlen, args.ic_perturb, args.fn)
