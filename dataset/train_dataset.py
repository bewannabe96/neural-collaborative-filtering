import numpy as np
import torch
from torch.utils import data


class TrainDataset(data.Dataset):
    @staticmethod
    def __create_negative_samples(item_number, pos_sparse_user, pos_sparse_item, sample_ratio):
        all_items = torch.arange(item_number, dtype=torch.int32) + 1

        negative_samples = torch.tensor([], dtype=torch.int32)

        for user_id in pos_sparse_user.unique():
            positive_items = pos_sparse_item[pos_sparse_user == user_id]
            negative_sample_size = len(positive_items) * sample_ratio

            possible_negative_items = all_items[torch.isin(all_items, positive_items, invert=True)]

            random_indices = torch.randperm(len(possible_negative_items))[:negative_sample_size]

            negative_items = possible_negative_items[random_indices].unsqueeze(dim=1)
            negative_items = torch.cat((torch.full((negative_items.size(0), 1), user_id), negative_items), dim=1)

            negative_samples = torch.cat((negative_samples, negative_items))

        return negative_samples[:, 0], negative_samples[:, 1]

    def __init__(self, user_number: int, item_number: int, sparse_user: np.ndarray, sparse_item: np.ndarray,
                 negative_sample_ratio):
        super(TrainDataset, self).__init__()

        assert len(sparse_user) == len(sparse_item), "Length of `sparse_user` and `sparse_item` must be equal."

        self.user_number = user_number
        self.item_number = item_number
        self.negative_sample_ratio = negative_sample_ratio

        self.pos_sparse_user = torch.from_numpy(sparse_user)
        self.pos_sparse_item = torch.from_numpy(sparse_item)
        self.p_len = len(self.pos_sparse_user)
        self.pos_sparse_label = torch.tensor([1.0] * self.p_len)

        self.sparse_user = self.pos_sparse_user
        self.sparse_item = self.pos_sparse_item
        self.len = self.p_len
        self.sparse_label = self.pos_sparse_label

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.sparse_user[idx], self.sparse_item[idx], self.sparse_label[idx]

    def regenerate_negative_samples(self):
        neg_sparse_user, neg_sparse_item = TrainDataset.__create_negative_samples(
            self.item_number, self.pos_sparse_user, self.pos_sparse_item, self.negative_sample_ratio)
        n_len = neg_sparse_user.size(0)
        neg_sparse_label = torch.tensor([0.0] * n_len)

        self.sparse_user = torch.cat((self.pos_sparse_user, neg_sparse_user), dim=0)
        self.sparse_item = torch.cat((self.pos_sparse_item, neg_sparse_item), dim=0)
        self.len = self.p_len + n_len
        self.sparse_label = torch.cat((self.pos_sparse_label, neg_sparse_label), dim=0)

    def get_sparsity(self):
        return 1 - (self.len / (self.user_number * self.item_number))
