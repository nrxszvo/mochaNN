import argparse
import os

import numpy as np
import torch

from config import get_config
from dataset import NCMDataModule
from ncm import NeuralChaosModule

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cp", required=True, help="checkpoint file")
parser.add_argument(
    "--outfn",
    default=None,
    help="prediction file name",
)


def main():
    args = parser.parse_args()
    cpfn = args.cp
    cfgfn = cpfn.replace("ckpt", "yml")
    outfn = args.outfn
    if outfn is None:
        outfn = cpfn.replace("ckpt", "npy")

    cfgyml = get_config(cfgfn)

    torch.set_default_dtype(
        torch.float32 if cfgyml.dtype == "float32" else torch.float64
    )

    datamodule = NCMDataModule(
        cfgyml.datafile,
        cfgyml.dtype,
        cfgyml.ntrain,
        cfgyml.nval,
        cfgyml.ntest,
        cfgyml.npts,
        cfgyml.input_size,
        cfgyml.H,
        cfgyml.batch_size,
        os.cpu_count() - 1,
    )

    ncm = NeuralChaosModule.load_from_checkpoint(args.cp)
    y_hat, y_true = ncm.predict(datamodule)

    np.save(
        outfn,
        {
            "config": cfgyml,
            "y_true": y_true.numpy(),
            "y_hat": y_hat.numpy(),
        },
        allow_pickle=True,
    )


if __name__ == "__main__":
    main()
