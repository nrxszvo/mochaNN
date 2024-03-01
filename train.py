from datetime import datetime
import argparse
import os
from config import get_config
import numpy as np
import torch
from dataset import NCMDataModule
from ncm import NeuralChaosModule

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default="cfg.yml", help="yaml config file")
parser.add_argument(
    "--save", default=False, action="store_true", help="save prediction data"
)
parser.add_argument(
    "--fn",
    default=datetime.now().strftime("%Y%m%d%H%M%S"),
    help="prediction file name",
)


def main():
    args = parser.parse_args()
    cfgyml = get_config(args.cfg)

    torch.set_default_dtype(
        torch.float32 if cfgyml.dtype == "float32" else torch.float64
    )
    step_size = np.load(cfgyml.datafile, allow_pickle=True).item()["ndim"]
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

    ncm = NeuralChaosModule(
        cfgyml.H,
        cfgyml.input_size,
        step_size,
        cfgyml.nhits_params,
        cfgyml.loss,
        cfgyml.max_steps,
        cfgyml.val_check_steps,
        cfgyml.lr_schedule_params,
        cfgyml.random_seed,
    )

    ncm.fit(datamodule)

    y_hat, y_true = ncm.predict(datamodule)

    if args.save:
        datafile = f"predictions/{args.fn}.npy"
        np.save(
            datafile,
            {
                "config": cfgyml,
                "y_true": y_true.numpy(),
                "y_hat": y_hat.numpy(),
            },
            allow_pickle=True,
        )


if __name__ == "__main__":
    main()
