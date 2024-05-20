import numpy as np
import torch
from torch.utils import data


class ImplicitSparseDataset(data.Dataset):
    @staticmethod
    def __create_negative_samples(M, N, sparse_Y_u, sparse_Y_i, sample_ratio):
        sparse_Y = [(u.item(), i.item()) for (u, i) in zip(sparse_Y_u, sparse_Y_i)]

        negative_samples = []
        for u, i in zip(sparse_Y_u, sparse_Y_i):
            for _ in range(sample_ratio):
                while True:
                    i = np.random.randint(N)
                    if (u, i) not in sparse_Y:
                        break
                negative_samples.append((u, i))

        negative_samples = torch.tensor(negative_samples)
        return negative_samples[:, 0], negative_samples[:, 1]

    def __init__(self, M, N, sparse_Y_u, sparse_Y_i, include_negative_samples=False, negative_sample_ratio=1):
        super(ImplicitSparseDataset, self).__init__()

        assert len(sparse_Y_u) == len(sparse_Y_i), "Length of `sparse_Y_u` and `sparse_Y_i` must be equal."

        self.len = len(sparse_Y_u)
        self.sparsity = 1 - (self.len / (M * N))

        self.Y_u = sparse_Y_u
        self.Y_i = sparse_Y_i
        self.Y_value = torch.tensor([1.0] * self.len)

        if include_negative_samples:
            negative_sample_size = M * negative_sample_ratio
            self.len = self.len + negative_sample_size

            n_Y_u, n_Y_i = ImplicitSparseDataset.__create_negative_samples(M, N, self.Y_u, self.Y_i,
                                                                           negative_sample_ratio)

            self.Y_u = torch.cat((self.Y_u, n_Y_u), dim=0)
            self.Y_i = torch.cat((self.Y_i, n_Y_i), dim=0)
            self.Y_value = torch.cat((self.Y_value, torch.tensor([0.0] * negative_sample_size)), dim=0)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.Y_u[idx], self.Y_i[idx], self.Y_value[idx]

    def get_sparsity(self):
        return self.sparsity
