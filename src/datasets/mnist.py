import numpy as np
from pandas import DataFrame, read_csv
import torch
from torch.utils.data import Dataset
from typing import Tuple


class CSVDataset(Dataset):
    def __init__(self, file: str):
        data: DataFrame = read_csv(file)
        self.images = data.drop(columns=["label"]).values.reshape(-1, 1, 28, 28)
        self.images = (self.images / 255.).astype(np.float32)
        self.labels = data["label"].values
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        img = self.images[index]
        lbl = self.labels[index]
        return img, lbl
