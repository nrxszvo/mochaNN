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
    "--save_path",
    default="outputs",
    help="folder for saving config and checkpoints",
)
parser.add_argument(
    "--outfn",
    default=datetime.now().strftime("%Y%m%d%H%M%S"),
    help="prediction file name",
)
parser.add_argument(
    "--ngpu", default=1, type=int, help="number of gpus per training trial"
)


def tune_hps(save_path, name, get_ncm, get_dm, cfgyml, datafile):
    def fit(config):
        nhits_params = {
            "n_stacks": config["n_stacks"],
            "n_pool_kernel_size": config["n_pool_kernel_size"],
            "n_freq_downsample": config["n_freq_downsample"],
            "n_blocks": config["n_blocks"],
            "mlp_layers": config["mlp_layers"],
            "mlp_width": config["mlp_width"],
        }
        lr_scheduler_params = {
            "lr": config["lr"],
            "name": cfgyml.lr_scheduler_params["name"],
            "gamma": config["gamma"],
            "patience": cfgyml.lr_scheduler_params["patience"],
            "threshold": config["threshold"],
            "min_lr": cfgyml.lr_scheduler_params["min_lr"],
        }
        ncm = get_ncm("it", nhits_params, lr_scheduler_params, config["batch_size"])
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
        num_workers=cfgyml.num_workers,
        use_gpu=True,
        resources_per_worker=cfgyml.resources_per_worker,
    )
    reporter = CLIReporter(
        metric_columns=["valid_loss", "training_iteration"],
    )
    run_config = RunConfig(
        name=name,
        progress_reporter=reporter,
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="valid_loss",
            checkpoint_score_order="min",
        ),
    )

    trainer = TorchTrainer(
        fit,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    lr_params = cfgyml.lr_scheduler_params
    nhits_params = cfgyml.nhits_params
    search_space = {
        "batch_size": tune.grid_search(cfgyml.batch_size),
        "lr": tune.grid_search(lr_params["lr"]),
        "gamma": tune.grid_search(lr_params["gamma"]),
        "threshold": tune.grid_search(lr_params["threshold"]),
        "min_lr": tune.grid_search(lr_params["min_lr"]),
        "n_stacks": tune.choice([nhits_params["n_stacks"]]),
        "n_pool_kernel_size": tune.grid_search(nhits_params["n_pool_kernel_size"]),
        "n_freq_downsample": tune.grid_search(nhits_params["n_freq_downsample"]),
        "n_blocks": tune.grid_search(nhits_params["n_blocks"]),
        "mlp_layers": tune.grid_search(nhits_params["mlp_layers"]),
        "mlp_width": tune.grid_search(nhits_params["mlp_width"]),
    }

    def dncreator(trial):
        return f"t-{trial.trial_id[:-3]}"

    tuner = tune.Tuner(
        trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="valid_loss",
            mode="min",
            num_samples=1,
            scheduler=scheduler,
            trial_dirname_creator=dncreator,
        ),
    )
    return tuner.fit()


def main():
    args = parser.parse_args()
    cfgyml = get_config(args.cfg)
    os.makedirs(args.save_path, exist_ok=True)
    shutil.copyfile(args.cfg, f"{args.save_path}/{args.outfn}.yml")

    torch.set_default_dtype(
        torch.float32 if cfgyml.dtype == "float32" else torch.float64
    )

    md_file = cfgyml.datafile
    if os.path.isdir(md_file):
        md_file = f"{md_file}/md.npy"
    step_size = np.load(md_file, allow_pickle=True).item()["ndim"]

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
            cfgyml.stride,
            cfgyml.spacing,
            batch_size,
            os.cpu_count() - 1,
        )

    def get_ncm(
        name,
        outdir,
        nhits_params,
        lr_scheduler_params,
        batch_size,
        strategy,
        devices,
    ):
        return NeuralChaosModule(
            name,
            outdir,
            cfgyml.H,
            cfgyml.input_size,
            step_size,
            cfgyml.stride,
            nhits_params,
            lr_scheduler_params,
            cfgyml.loss,
            cfgyml.max_steps,
            cfgyml.val_check_steps,
            cfgyml.random_seed,
            batch_size,
            strategy,
            devices,
        )

    if args.tune:
        datafile = os.path.abspath(cfgyml.datafile)
        results = tune_hps(
            args.save_path,
            args.outfn,
            get_ncm,
            get_datamodule,
            cfgyml,
            datafile,
        )
        print("best hps found:")
        config = results.get_best_result().config["train_loop_config"]
        for k, v in config.items():
            print(f"\t{k}:\t{v}")

    else:
        ncm = get_ncm(
            args.outfn,
            f"{args.save_path}/models",
            cfgyml.nhits_params,
            cfgyml.lr_scheduler_params,
            cfgyml.batch_size,
            cfgyml.strategy,
            args.ngpu,
        )
        dm = get_datamodule(cfgyml.batch_size, cfgyml.datafile)
        ncm.fit(dm)
        config = cfgyml


if __name__ == "__main__":
    main()
