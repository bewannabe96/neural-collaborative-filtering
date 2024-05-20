import pandas as pd
import torch
import numpy as np
from validation_dataset import ValidationDataset
from implicit_sparse_dataset import ImplicitSparseDataset


def create_dataset(train_csv, val_csv, user_number, item_number, negative_sample_ratio):
    train_dataset = pd.read_csv(train_csv, dtype={0: np.int32, 1: np.int32}, header=None)
    train_dataset = torch.tensor(train_dataset.values)
    train_dataset = ImplicitSparseDataset(user_number, item_number, train_dataset[:, 0], train_dataset[:, 1],
                                          include_negative_samples=True, negative_sample_ratio=negative_sample_ratio)

    val_dataset = ValidationDataset(val_csv)

    return train_dataset, val_dataset
