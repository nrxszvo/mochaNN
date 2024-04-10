import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as L


class NCMDataset(Dataset):
    def __init__(self, series, input_size, h, edges=None, edge_cdf=None):
        super().__init__()
        self.h = h
        self.input_size = input_size
        self.window_size = input_size + h
        self.series = series
        nseries, npts, ndim = self.series.shape
        self.nseries = nseries
        self.npts = npts
        self.win_per_series = self.npts - self.window_size + 1
        self.edges = edges
        self.edge_cdf = edge_cdf
        self._categorize_datapoints()

    def __len__(self):
        return self.nseries * self.win_per_series

    def __getitem__(self, idx):
        if len(self.categories) > 0:
            cumprob = np.random.random()
            for i in range(len(self.edge_cdf)):
                if cumprob < self.edge_cdf[i]:
                    category = self.categories[i]
                    break

            coord_idx = np.random.choice(category.shape[0], 1)[0]
            s_idx, p_idx = category[coord_idx]

            # randomly align p_idx within the target window if possible
            r = np.random.random()
            if p_idx > self.win_per_series + self.input_size:
                w_idx = self.win_per_series - 1 - int(r * (self.npts - p_idx))
            elif p_idx < self.window_size:
                w_idx = int(r * p_idx)
            else:
                w_idx = p_idx - 1 - int(self.h * r) - self.input_size
        else:
            s_idx = idx // self.win_per_series
            w_idx = idx - (s_idx * self.win_per_series)

        window = self.series[s_idx, w_idx : w_idx + self.window_size]
        return {
            "input": window[: self.input_size].reshape(-1),
            "target": window[self.input_size :].reshape(-1),
        }

    def _categorize_datapoints(self):
        self.categories = []
        if self.edges is not None:
            dfo = np.linalg.norm(self.series, axis=2)
            b = 0
            for e in self.edges:
                self.categories.append(np.argwhere((dfo >= b) & (dfo < e)))
                b = e


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
        edges=None,
        edge_cdf=None,
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
        self.edges = edges
        self.edge_cdf = edge_cdf
        del self._data["data"]

    def setup(self, stage):
        if stage == "fit":
            self.trainset = NCMDataset(
                self.series[: self.ntrain, : self.npts],
                self.input_size,
                self.h,
                self.edges,
                self.edge_cdf,
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
