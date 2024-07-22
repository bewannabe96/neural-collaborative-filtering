import torch
from torch import nn


class MF(nn.Module):
    def __init__(self, M, N, embedding_dim):
        super(MF, self).__init__()

        self.M = M
        self.N = N

        self.embedding_dim = embedding_dim

        self.P = nn.Embedding(num_embeddings=self.M, embedding_dim=self.embedding_dim)
        self.Q = nn.Embedding(num_embeddings=self.N, embedding_dim=self.embedding_dim)

        self.__init_weights()

    def __init_weights(self):
        nn.init.normal_(self.P.weight, std=.01)
        nn.init.normal_(self.Q.weight, std=.01)

    def forward(self, user_id, item_id):
        p_u = self.P(user_id)
        q_u = self.Q(item_id)

        out = torch.multiply(p_u, q_u)
        out = torch.sum(out, dim=-1)

        return out.view(-1, 1)
