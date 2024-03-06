import argparse
import os
import shutil
from datetime import datetime
import lightning as L
import numpy as np
import torch
from ncmdataset import NCMDataModule
from ncm import NeuralChaosModule
from ray import tune
from ray.tune import CLIReporter
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from config import get_config

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default="cfg.yml", help="yaml config file")
parser.add_argument(
    "--tune", action="store_true", default=False, help="run ray hyperparameter tuning"
)
parser.add_argument(
    "--save", default=False, action="store_true", help="save prediction data"
)
parser.add_argument(
    "--save_path",
    default="outputs",
    help="folder for saving predictions",
)
parser.add_argument(
    "--outfn",
    default=datetime.now().strftime("%Y%m%d%H%M%S"),
    help="prediction file name",
)


def tune_hps(save_path, name, get_ncm, get_dm, cfgyml, datafile):
    def fit(config):
        nhits_params = {
            "n_stacks": config["n_stacks"],
            "n_pool_kernel_size": config["n_pool_kernel_size"],
            "n_freq_downsample": config["n_freq_downsample"],
            "n_blocks": config["n_blocks"],
            "mlp_units": config["mlp_units"],
        }
        lr_scheduler_params = {
            "lr": config["lr"],
            "name": cfgyml.lr_scheduler_params["name"],
            "gamma": config["gamma"],
            "patience": cfgyml.lr_scheduler_params["patience"],
            "threshold": config["threshold"],
            "min_lr": cfgyml.lr_scheduler_params["min_lr"],
        }
        ncm = get_ncm(nhits_params, lr_scheduler_params)
        trainer = L.Trainer(
            callbacks=[RayTrainReportCallback()],
            strategy=RayDDPStrategy(),
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=False,
            enable_checkpointing=False,
            **ncm.trainer_kwargs,
        )
        trainer = prepare_trainer(trainer)
        dm = get_dm(config["batch_size"], datafile)
        trainer.fit(ncm, datamodule=dm)

    scheduler = ASHAScheduler(
        reduction_factor=cfgyml.reduction_factor,
        max_t=cfgyml.max_steps,
        grace_period=cfgyml.min_steps,
        brackets=cfgyml.brackets,
    )
    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,
        resources_per_worker={"CPU": 1, "GPU": 1},
    )
    reporter = CLIReporter(
        metric_columns=["valid_loss", "training_iteration"],
    )
    run_config = RunConfig(
        name=name,
        progress_reporter=reporter,
        storage_path=os.path.abspath(f"{save_path}/ray_results"),
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="valid_loss",
            checkpoint_score_order="min",
        ),
    )

    trainer = TorchTrainer(fit, scaling_config=scaling_config, run_config=run_config)
    lr_params = cfgyml.lr_scheduler_params
    nhits_params = cfgyml.nhits_params
    search_space = {
        "batch_size": tune.grid_search(cfgyml.batch_size),
        "lr": tune.grid_search(lr_params["lr"]),
        "gamma": tune.grid_search(lr_params["gamma"]),
        "threshold": tune.grid_search(lr_params["threshold"]),
        "n_stacks": tune.choice([nhits_params["n_stacks"]]),
        "n_pool_kernel_size": tune.grid_search(nhits_params["n_pool_kernel_size"]),
        "n_freq_downsample": tune.grid_search(nhits_params["n_freq_downsample"]),
        "n_blocks": tune.grid_search(nhits_params["n_blocks"]),
        "mlp_units": tune.grid_search(nhits_params["mlp_units"]),
    }
    tuner = tune.Tuner(
        trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="valid_loss", mode="min", num_samples=1, scheduler=scheduler
        ),
    )
    return tuner.fit()


def main():
    args = parser.parse_args()
    cfgyml = get_config(args.cfg)
    os.makedirs(args.save_path, exist_ok=True)
    if args.save:
        shutil.copyfile(args.cfg, f"{args.save_path}/{args.outfn}.yml")

    torch.set_default_dtype(
        torch.float32 if cfgyml.dtype == "float32" else torch.float64
    )
    step_size = np.load(cfgyml.datafile, allow_pickle=True).item()["ndim"]

    def get_datamodule(batch_size, datafile):
        return NCMDataModule(
            datafile,
            cfgyml.dtype,
            cfgyml.ntrain,
            cfgyml.nval,
            cfgyml.ntest,
            cfgyml.npts,
            cfgyml.input_size,
            cfgyml.H,
            batch_size,
            os.cpu_count() - 1,
        )

    def get_ncm(nhits_params, lr_scheduler_params):
        return NeuralChaosModule(
            args.outfn,
            cfgyml.H,
            cfgyml.input_size,
            step_size,
            nhits_params,
            cfgyml.loss,
            cfgyml.max_steps,
            cfgyml.val_check_steps,
            lr_scheduler_params,
            cfgyml.random_seed,
        )

    if args.tune:
        datafile = os.path.abspath(cfgyml.datafile)
        results = tune_hps(
            args.save_path, args.outfn, get_ncm, get_datamodule, cfgyml, datafile
        )
        print("best hps found:")
        for k, v in results.get_best_result().config["train_loop_config"].items():
            print(f"\t{k}:\t{v}")

        if args.save:
            dm = get_datamodule(cfgyml.batch_size[0], datafile)
            cp_obj = results.get_best_result().checkpoint
            cp = cp_obj.path + "/checkpoint.ckpt"
            model_path = f"{args.save_path}/models"
            os.makedirs(model_path, exist_ok=True)
            shutil.copyfile(cp, f"{model_path}/{args.outfn}.ckpt")
            breakpoint()
            ncm = NeuralChaosModule.load_from_checkpoint(cp)
    else:
        ncm = get_ncm(cfgyml.nhits_params, cfgyml.lr_scheduler_params)
        dm = get_datamodule(cfgyml.batch_size, cfgyml.datafile)
        ncm.fit(dm)

    if args.save:
        y_hat, y_true = ncm.predict(dm)
        datadir = f"{args.save_path}/predictions"
        os.makedirs(datadir, exist_ok=True)
        np.save(
            f"{datadir}/{args.outfn}.npy",
            {
                "config": cfgyml,
                "y_true": y_true.numpy(),
                "y_hat": y_hat.numpy(),
            },
            allow_pickle=True,
        )


if __name__ == "__main__":
    main()
