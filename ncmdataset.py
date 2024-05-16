import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as L
import os


class NCMDataset(Dataset):
    def __init__(self, series, input_size, h, stride=1):
        super().__init__()
        self.h = h
        self.input_size = input_size
        self.window_size = input_size + h
        self.stride = stride
        self.series = series
        nseries, npts, ndim = self.series.shape
        self.nseries = nseries
        self.npts = npts
        self.win_per_series = (self.npts - self.window_size) // self.stride + 1

    def __len__(self):
        return self.nseries * self.win_per_series

    def __getitem__(self, idx):
        s_idx = idx // self.win_per_series
        w_idx = self.stride * (idx - (s_idx * self.win_per_series))

        window = self.series[s_idx, w_idx : w_idx + self.window_size].copy()
        return {
            "input": window[: self.input_size].reshape(-1),
            "target": window[self.input_size :].reshape(-1),
        }


class NCMDataModule(L.LightningDataModule):
    def __init__(
        self,
        datafile,
        dtype_str,
        ntrain,
        nval,
        ntest,
        npts,
        input_size,
        h,
        stride,
        spacing,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.datafile = datafile
        self.dtype = np.float32 if dtype_str == "float32" else np.float64
        self.ntrain = ntrain
        self.nval = nval
        self.ntest = ntest
        self.npts = npts
        self.input_size = input_size
        self.h = h
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        if os.path.isdir(self.datafile):
            md = np.load(f"{self.datafile}/md.npy", allow_pickle=True).item()
            solns = np.memmap(
                f"{self.datafile}/solutions.npy",
                mode="r",
                dtype="float32",
                shape=md["shape"],
            )
            self.dt = md["dt"]
        else:
            data = np.load(self.datafile, allow_pickle=True).item()
            solns = data["solutions"]
            self.dt = data["dt"]
        self.series = solns[:, ::spacing].astype(self.dtype, copy=False)

    def setup(self, stage):
        if stage == "fit":
            self.trainset = NCMDataset(
                self.series[: self.ntrain, : self.npts],
                self.input_size,
                self.h,
                self.stride,
            )
        if stage in ["fit", "validate"]:
            self.valset = NCMDataset(
                self.series[self.ntrain : self.ntrain + self.nval, : self.npts],
                self.input_size,
                self.h,
                self.stride,
            )

        if stage in ["test", "predict"]:
            self.testset = NCMDataset(
                self.series[
                    self.ntrain + self.nval : self.ntrain + self.nval + self.ntest,
                    : self.npts,
                ],
                self.input_size,
                self.h,
                self.stride,
            )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            worker_init_fn=lambda id: np.random.seed(id),
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.testset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return self.predict_dataloader()
