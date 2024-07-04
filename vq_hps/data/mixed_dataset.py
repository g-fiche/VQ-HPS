"""Adapted from https://github.com/pixelite1201/BEDLAM/blob/master/train/dataset/mixed_dataset.py"""

import torch
import numpy as np
import sys

sys.path.append("./")
from vq_hps.data.dataset_hmr import DatasetHMR


class MixedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file: str,
        list_ratios: list = None,
        augment: bool = True,
        flip: bool = True,
        proportion: float = 1,
    ):
        dataset_file = open(file, "r")
        data = dataset_file.read()
        self.dataset_list = data.split("\n")

        self.proportion = proportion

        if list_ratios is not None:
            self.dataset_ratios = list_ratios
        else:
            self.dataset_ratios = [1 / len(self.dataset_list)] * len(self.dataset_list)

        assert len(self.dataset_list) == len(
            self.dataset_ratios
        ), "Number of datasets and ratios should be equal"

        print(len(self.dataset_list))

        self.datasets = [
            DatasetHMR(
                ds,
                augment=augment,
                flip=flip,
                proportion=self.proportion,
            )
            for ds in self.dataset_list
        ]
        self.length = max([len(ds) for ds in self.datasets])

        self.partition = []

        for idx, (ds_name, ds_ratio) in enumerate(
            zip(self.dataset_list, self.dataset_ratios)
        ):
            r = ds_ratio
            self.partition.append(r)

        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(len(self.datasets)):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
