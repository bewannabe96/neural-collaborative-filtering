import numpy as np
import torch
from torch.utils import data


class TrainDataset(data.Dataset):
    @staticmethod
    def __create_negative_samples(sparse_user: torch.Tensor, sparse_neg_items: torch.Tensor, sample_ratio: int):
        num_rows, num_cols = sparse_neg_items.shape

        sparse_neg_user = sparse_user.unsqueeze(1).repeat(1, sample_ratio).reshape(-1)

        sampled_rows = []
        for row in sparse_neg_items:
            indices = torch.randperm(num_cols)[:sample_ratio]
            sampled_row = row[indices]
            sampled_rows.append(sampled_row)
        sparse_neg_sample_items = torch.stack(sampled_rows).reshape(-1)

        return sparse_neg_user, sparse_neg_sample_items

    def __init__(self, sparse_user: np.ndarray, sparse_pos_item: np.ndarray, sparse_neg_items: np.ndarray,
                 negative_sample_ratio: int):
        super(TrainDataset, self).__init__()

        assert len(sparse_user) == len(sparse_pos_item), \
            "Length of `sparse_user` and `sparse_pos_item` must be equal."
        assert len(sparse_user) == len(sparse_neg_items), \
            "Length of `sparse_user` and `sparse_neg_items` must be equal."

        self.negative_sample_ratio = negative_sample_ratio

        self.original_size = len(sparse_user)
        self.sparse_original_user = torch.from_numpy(sparse_user)
        self.sparse_original_pos_item = torch.from_numpy(sparse_pos_item)
        self.sparse_original_neg_items = torch.from_numpy(sparse_neg_items)

        self.size = self.original_size
        self.sparse_user = self.sparse_original_user
        self.sparse_item = self.sparse_original_pos_item
        self.sparse_label = torch.tensor([1.0] * self.original_size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.sparse_user[idx], self.sparse_item[idx], self.sparse_label[idx]

    def regenerate_negative_samples(self):
        sparse_neg_user, sparse_neg_item = TrainDataset.__create_negative_samples(
            self.sparse_original_user, self.sparse_original_neg_items, self.negative_sample_ratio
        )

        neg_size = sparse_neg_user.size(0)

        self.size = self.original_size + neg_size
        self.sparse_user = torch.cat((self.sparse_original_user, sparse_neg_user), dim=0)
        self.sparse_item = torch.cat((self.sparse_original_pos_item, sparse_neg_item), dim=0)
        self.sparse_label = torch.cat((
            torch.tensor([1.0] * self.original_size), torch.tensor([0.0] * sparse_neg_user.size(0))), dim=0
        )
