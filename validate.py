import argparse
import os

import numpy as np
import torch

from config import get_config
from ncmdataset import NCMDataModule
from ncm import NeuralChaosModule


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", required=True, help="yaml config file")
parser.add_argument("--cp", required=True, help="checkpoint file")
parser.add_argument(
    "--npy",
    help="optional npy file or directory of files containing series to predict; default is to use the test set from the original dataset",
    default=None,
)


def main():
    args = parser.parse_args()
    cpfn = args.cp
    cfgfn = args.cfg
    cfgyml = get_config(cfgfn)

    torch.set_default_dtype(
        torch.float32 if cfgyml.dtype == "float32" else torch.float64
    )

    if isinstance(cfgyml.batch_size, list):
        batch_size = cfgyml.batch_size[0]
    else:
        batch_size = cfgyml.batch_size

    stride = getattr(cfgyml, "stride", 1)
    step_size = np.load(cfgyml.datafile, allow_pickle=True).item()["ndim"]
    ncm = NeuralChaosModule.load_from_checkpoint(
        args.cp,
        name="validator",
        outdir="outputs/models",
        h=cfgyml.H,
        input_size=cfgyml.input_size,
        step_size=step_size,
        stride=stride,
        model_params=cfgyml.nhits_params,
        lr_scheduler_params=cfgyml.lr_scheduler_params,
    )
    if args.npy is None:
        npy = cfgyml.datafile
        ntrain = cfgyml.ntrain
        nval = cfgyml.nval
        ntest = cfgyml.ntest
    else:
        npy = args.npy
        ntrain = 0
        ntest = 0
        nval = np.load(npy, allow_pickle=True).item()["solutions"].shape[0]
        batch_size = min(batch_size, nval)

    datamodule = NCMDataModule(
        npy,
        cfgyml.dtype,
        ntrain,
        nval,
        ntest,
        cfgyml.npts,
        cfgyml.input_size,
        cfgyml.H,
        stride,
        batch_size,
        os.cpu_count() - 1,
    )

    ncm.validate(datamodule)


if __name__ == "__main__":
    main()
