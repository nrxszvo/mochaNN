import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as L


class NCMDataset(Dataset):
    def __init__(self, series, input_size, h):
        super().__init__()
        self.input_size = input_size
        self.window_size = input_size + h
        self.series = series
        nseries, npts, ndim = self.series.shape
        self.nseries = nseries
        self.win_per_series = npts - self.window_size + 1

    def __len__(self):
        return self.nseries * self.win_per_series

    def __getitem__(self, idx):
        s_idx = idx // self.win_per_series
        w_idx = idx - (s_idx * self.win_per_series)
        window = self.series[s_idx, w_idx : w_idx + self.window_size]
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
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._data = np.load(self.datafile, allow_pickle=True).item()
        self.series = self._data["data"].astype(self.dtype, copy=False)
        del self._data["data"]

    def setup(self, stage):
        if stage == "fit":
            self.trainset = NCMDataset(
                self.series[: self.ntrain, : self.npts],
                self.input_size,
                self.h,
            )
            self.valset = NCMDataset(
                self.series[self.ntrain : self.ntrain + self.nval, : self.npts],
                self.input_size,
                self.h,
            )

        if stage in ["test", "predict"]:
            self.testset = NCMDataset(
                self.series[
                    self.ntrain + self.nval : self.ntrain + self.nval + self.ntest,
                    : self.npts,
                ],
                self.input_size,
                self.h,
            )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
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
