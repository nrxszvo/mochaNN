import argparse
import torch
from nhits import NHITS
from config import get_config
import numpy as np
import os


class Wrapper(torch.nn.Module):
    def __init__(self, H, L, nhits_params, state_dict, device):
        super().__init__()
        self.model = NHITS(H, L, **nhits_params)
        self.load_state_dict(state_dict)
        self.to(device)

    def forward(self, inputs):
        return self.model(inputs)


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Generate predictions auto-regressively",
)
parser.add_argument(
    "--cfg", required=True, help="yaml config file that was used to train the model"
)
parser.add_argument("--cp", required=True, help="checkpoint file")
parser.add_argument(
    "--outfn",
    default=None,
    help="prediction file name",
)
parser.add_argument(
    "--inc",
    default=None,
    type=int,
    help="number of predicted samples to save per prediction; 'None' means use the entire horizon",
)
parser.add_argument("--gpu", action="store_true", help="use gpu")
parser.add_argument(
    "--npy",
    default=None,
    help="optional npy file containing trajectories to predict (overrides datafile from cfg)",
)
args = parser.parse_args()

cfgyml = get_config(args.cfg)
spacing = getattr(cfgyml, "spacing", 1)

cp = torch.load(args.cp, map_location="cpu")

if args.npy is not None:
    dset = np.load(args.npy, allow_pickle=True).item()
    test_series = dset["solutions"]
else:
    dset = np.load(cfgyml.datafile, allow_pickle=True).item()
    test_series = dset["solutions"][cfgyml.ntrain + cfgyml.nval :]

test_series = test_series[:, ::spacing]
ntest, n_total_pts, ndim = test_series.shape
H = cfgyml.H
L = cfgyml.input_size
npts = n_total_pts - L
Hf = H * ndim
Lf = L * ndim

device = "cuda" if args.gpu else "cpu"

model = Wrapper(Hf, Lf, cfgyml.nhits_params, cp["state_dict"], device)
model.eval()

inputs = torch.from_numpy(test_series[:, :L].reshape(ntest, -1)).type(torch.float32)
outputs = torch.empty((ntest, n_total_pts * ndim)).to(device)

outputs[:, :Lf] = inputs
idx = Lf

os.makedirs(args.outfn, exist_ok=True)
yt_map = np.memmap(
    f"{args.outfn}/ytrue.npy", mode="w+", dtype="float32", shape=(ntest, npts, ndim)
)
yt_map[:] = test_series[:, L:]

yh_map = np.memmap(
    f"{args.outfn}/yhat.npy", mode="w+", dtype="float32", shape=(ntest, npts, ndim)
)

if args.inc is not None:
    inc = args.inc * ndim
else:
    inc = Hf
while idx < n_total_pts * ndim:
    print(idx, end="\r")
    with torch.no_grad():
        hpred = model(outputs[:, idx - Lf : idx])
        rem = min(inc, n_total_pts * ndim - idx)
        outputs[:, idx : idx + rem] = hpred[:, :rem]
    outidx = idx // ndim - L
    outwin = outputs[:, idx : idx + rem].reshape(ntest, -1, ndim).cpu().numpy()
    yh_map[:, outidx : outidx + outwin.shape[1]] = outwin
    idx += inc

print()
np.save(
    f"{args.outfn}/md.npy",
    {"mode": "ar", "dt": dset["dt"], "config": cfgyml, "shape": (ntest, npts, ndim)},
    allow_pickle=True,
)
