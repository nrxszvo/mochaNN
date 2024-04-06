import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from nhits import NHITS


class NeuralChaosModule(L.LightningModule):
    def __init__(
        self,
        name,
        h,
        input_size,
        step_size,
        model_params,
        lr_scheduler_params=None,
        loss="MAE",
        max_steps=100,
        val_check_steps=100,
        random_seed=0,
        batch_size=32,
        strategy="auto",
        devices=1,
    ):
        super().__init__()
        L.seed_everything(random_seed, workers=True)
        self.save_hyperparameters(
            model_params | lr_scheduler_params | {"batch_size": batch_size}
        )
        self.model = NHITS(h * step_size, input_size * step_size, **model_params)
        self.step_size = step_size
        self.h = h
        self.input_size = input_size
        if loss == "MAE":
            self.loss = nn.L1Loss()
        else:
            raise Exception
        self.val_check_steps = val_check_steps
        self.lr_scheduler_params = lr_scheduler_params
        self.max_steps = max_steps
        logger = TensorBoardLogger(".", name="L", version=name)
        self.trainer_kwargs = {
            "logger": logger,
            "max_steps": max_steps,
            "val_check_interval": min(val_check_steps, max_steps),
            "check_val_every_n_epoch": None,
            "strategy": strategy,
            "devices": devices,
        }

    def configure_optimizers(self):
        lr = self.lr_scheduler_params["lr"]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        name = self.lr_scheduler_params["name"]
        if name == "ReduceLROnPlateau":
            gamma = self.lr_scheduler_params["gamma"]
            threshold = self.lr_scheduler_params["threshold"]
            patience = self.lr_scheduler_params["patience"]
            min_lr = self.lr_scheduler_params["min_lr"]
            scheduler = ReduceLROnPlateau(
                optimizer=optimizer,
                factor=gamma,
                threshold=threshold,
                patience=patience,
                min_lr=min_lr,
            )
            freq = self.val_check_steps
        elif name == "StepLR":
            gamma = self.lr_scheduler_params["gamma"]
            num_decay = self.lr_scheduler_params["num_lr_decays"]
            step_size = int(self.max_steps / num_decay)
            scheduler = StepLR(
                optimizer=optimizer,
                step_size=step_size,
                gamma=gamma,
            )
            freq = 1
        config = {
            "scheduler": scheduler,
            "frequency": freq,
            "interval": "step",
            "monitor": "valid_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": config}

    def forward(self, inputs, target=None):
        insample_y = inputs[0]
        return self.model(insample_y)

    def training_step(self, batch, batch_idx):
        pred = self([batch["input"]])
        loss = self.loss(pred, batch["target"])
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self([batch["input"]])
        valid_loss = self.loss(pred, batch["target"])

        if torch.isnan(valid_loss):
            raise Exception("Loss is NaN, training stopped.")

        self.log("valid_loss", valid_loss, prog_bar=True, sync_dist=True)
        return valid_loss

    def predict_step(self, batch, batch_idx):
        return self([batch["input"]])

    def fit(self, datamodule, outdir, cpname):
        callbacks = [
            TQDMProgressBar(),
            ModelCheckpoint(dirpath=outdir, filename=cpname, save_weights_only=True),
        ]
        trainer = L.Trainer(callbacks=callbacks, **self.trainer_kwargs)
        trainer.fit(self, datamodule=datamodule)

    def predict(self, datamodule):
        tkargs = self.trainer_kwargs
        tkargs["strategy"] = "auto"
        tkargs["devices"] = 1
        trainer = L.Trainer(callbacks=[TQDMProgressBar()], **tkargs)
        pred = trainer.predict(self, datamodule)
        yt_raw = torch.from_numpy(datamodule.testset.series)

        nseries, npts, ndim = yt_raw.shape
        nwin = npts - (self.h + self.input_size) + 1

        yt_raw = yt_raw[:, self.input_size :, :]
        unfold = torch.nn.Unfold((self.h, 1))
        y_true = unfold(yt_raw.permute(0, 2, 1).unsqueeze(-1)).permute(0, 2, 1)
        y_true = y_true.reshape(nseries, nwin, ndim, self.h).permute(0, 1, 3, 2)
        y_hat = torch.cat(pred, dim=0)
        y_hat = y_hat.reshape(nseries, nwin, self.h, ndim)

        return y_hat, y_true
