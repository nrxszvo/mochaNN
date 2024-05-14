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
parser.add_argument(
    "--outfn",
    default=None,
    help="prediction file name",
)
parser.add_argument(
    "--stride", default=1, type=int, help="stride for computing windows"
)


def main():
    args = parser.parse_args()
    cpfn = args.cp
    cfgfn = args.cfg
    outfn = args.outfn
    if outfn is None:
        outfn = cpfn.replace("ckpt", "npy")

    cfgyml = get_config(cfgfn)

    torch.set_default_dtype(
        torch.float32 if cfgyml.dtype == "float32" else torch.float64
    )

    if isinstance(cfgyml.batch_size, list):
        batch_size = cfgyml.batch_size[0]
    else:
        batch_size = cfgyml.batch_size

    step_size = np.load(cfgyml.datafile, allow_pickle=True).item()["ndim"]
    ncm = NeuralChaosModule.load_from_checkpoint(
        args.cp,
        name="predictor",
        outdir="outputs/models",
        h=cfgyml.H,
        input_size=cfgyml.input_size,
        step_size=step_size,
        stride=args.stride,
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
        nval = 0
        ntest = np.load(npy, allow_pickle=True).item()["solutions"].shape[0]
        batch_size = min(batch_size, ntest)

    outdir = os.path.dirname(outfn)
    outname = os.path.splitext(os.path.basename(outfn))[0]
    os.makedirs(f"{outdir}/{outname}", exist_ok=True)

    datamodule = NCMDataModule(
        npy,
        cfgyml.dtype,
        ntrain,
        nval,
        ntest,
        cfgyml.npts,
        cfgyml.input_size,
        cfgyml.H,
        getattr(cfgyml, "stride", 1),
        getattr(cfgyml, "spacing", 1),
        batch_size,
        os.cpu_count() - 1,
    )

    y_hat, y_true = ncm.predict(datamodule)

    yhatmap = np.memmap(
        f"{outdir}/{outname}/{outname}_yhat.npy",
        dtype="float32",
        mode="w+",
        shape=y_hat.shape,
    )
    yhatmap[:] = y_hat
    yhatmap.flush()
    ytruemap = np.memmap(
        f"{outdir}/{outname}/{outname}_ytrue.npy",
        dtype="float32",
        mode="w+",
        shape=y_true.shape,
    )
    ytruemap[:] = y_true
    ytruemap.flush()

    np.save(
        f"{outdir}/{outname}/{outname}_md.npy",
        {
            "stride": getattr(cfgyml, "stride", 1),
            "config": cfgyml,
            "shape": y_hat.shape,
            "npts": y_true.shape[1],
        },
        allow_pickle=True,
    )


if __name__ == "__main__":
    main()
