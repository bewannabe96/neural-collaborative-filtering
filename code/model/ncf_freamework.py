import torch
from torch import nn


class NCFFramework(nn.Module):
    @staticmethod
    def __create_layer(in_size, dropout_prob=0.5):
        out_size = in_size // 2
        return nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ), out_size

    def __init__(self, user_number, item_number, ncfl_layer_number, ncfl_out_size, dropout_prob=0.5):
        super(NCFFramework, self).__init__()

        self.M = user_number
        self.N = item_number

        assert ncfl_layer_number >= 1, "Neural CF layers should have at least one layer."
        self.ncfl_layer_number = ncfl_layer_number  # Number of layers in 'neural collaborative filtering layers'

        self.ncfl_in_size = ((2 ** self.ncfl_layer_number) * ncfl_out_size)
        self.ncfl_out_size = ncfl_out_size  # 'neural collaborative filtering layers' output dimension

        self.embedding_dim = self.ncfl_in_size // 2

        self.P = nn.Embedding(num_embeddings=self.M, embedding_dim=self.embedding_dim)
        self.Q = nn.Embedding(num_embeddings=self.N, embedding_dim=self.embedding_dim)

        last_out_size = self.ncfl_in_size
        layers = []
        for i in range(self.ncfl_layer_number):
            layer, last_out_size = NCFFramework.__create_layer(last_out_size, dropout_prob)
            layers.append(layer)
        self.phi_X = nn.Sequential(*layers)

        self.phi_out = nn.Linear(last_out_size, 1)

        self.__init_weights()

    def __init_weights(self):
        nn.init.normal_(self.P.weight, std=.01)
        nn.init.normal_(self.Q.weight, std=.01)

        for phi in self.phi_X:
            for phi_layer in phi:
                if isinstance(phi_layer, nn.Linear):
                    nn.init.xavier_uniform_(phi_layer.weight)
                    phi_layer.bias.data.zero_()

        nn.init.kaiming_uniform_(self.phi_out.weight, nonlinearity="sigmoid")
        self.phi_out.bias.data.zero_()

    def forward(self, user_id, item_id):
        p_u = self.P(user_id)
        q_i = self.Q(item_id)

        phi_1_result = torch.cat((p_u, q_i), -1)
        phi_X_result = self.phi_X(phi_1_result)
        return self.phi_out(phi_X_result)