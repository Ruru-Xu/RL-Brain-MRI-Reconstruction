import argparse
import pathlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .data_loading import MRIDataset


class MRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: argparse.ArgumentParser,
    ):
        super().__init__()
        self.config = config
        self.setup()

    def setup(self, stage=None):
        self.test_dataset = MRIDataset(data_path=pathlib.Path(self.config.test_path))

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.test_batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
        )
