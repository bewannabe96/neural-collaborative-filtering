import numpy as np
import torch
from torch.utils import data


class ValidationDataset(data.Dataset):
    def __init__(self, sparse_user: np.ndarray, sparse_pos_item: np.ndarray, sparse_neg_items: np.ndarray):
        super(ValidationDataset, self).__init__()

        assert len(sparse_user) == len(sparse_pos_item),\
            "Length of `sparse_user` and `sparse_pos_item` must be equal."
        assert len(sparse_user) == len(sparse_neg_items),\
            "Length of `sparse_user` and `sparse_neg_items` must be equal."

        self.size = len(sparse_user)
        self.sparse_user = torch.from_numpy(sparse_user)
        self.pos_sparse_item = torch.from_numpy(sparse_pos_item)
        self.sparse_neg_items = torch.from_numpy(sparse_neg_items)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.sparse_user[idx], self.pos_sparse_item[idx], self.sparse_neg_items[idx]
