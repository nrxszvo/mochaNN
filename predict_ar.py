import argparse
import torch
from nhits import NHITS
from config import get_config
import numpy as np


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
cp = torch.load(args.cp, map_location="cpu")

if args.npy is not None:
    test_series = np.load(args.npy, allow_pickle=True).item()["solutions"]
else:
    series = np.load(cfgyml.datafile, allow_pickle=True).item()["solutions"]
    test_series = series[cfgyml.ntrain + cfgyml.nval :]

ntest, npts, ndim = test_series.shape
H = cfgyml.H
L = cfgyml.input_size
Hf = H * ndim
Lf = L * ndim

device = "cuda" if args.gpu else "cpu"

model = Wrapper(Hf, Lf, cfgyml.nhits_params, cp["state_dict"], device)
model.eval()

inputs = torch.from_numpy(test_series[:, :L].reshape(ntest, -1)).type(torch.float32)
outputs = torch.empty((ntest, npts * ndim)).to(device)

outputs[:, :Lf] = inputs
idx = Lf

if args.inc is not None:
    inc = args.inc * ndim
else:
    inc = Hf
while idx < npts * ndim:
    print(idx, end="\r")
    with torch.no_grad():
        hpred = model(outputs[:, idx - Lf : idx])
        rem = min(inc, npts * ndim - idx)
        outputs[:, idx : idx + rem] = hpred[:, :rem]
    idx += inc
print()
np.save(
    args.outfn,
    {"config": cfgyml, "y_true": test_series, "y_hat": outputs.cpu().numpy()},
    allow_pickle=True,
)
